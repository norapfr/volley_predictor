import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.ingestion.fivb_vis_crawler import CrawlState, FivbVisCrawler, is_in_scope
from src.ingestion.fivb_vis_source import (
    _extract_date_only,
    canonical_competition_name,
    iter_fivb_vis_raw_matches,
    resolve_gender,
)


class FakeVisClient:
    """Sustituye a FivbVisClient en los tests: no hace red, devuelve datos fijos."""

    def __init__(self, tournaments: List[Dict], matches_by_tournament: Dict[str, List[Dict]]):
        self.tournaments = tournaments
        self.matches_by_tournament = matches_by_tournament
        self.calls: List[str] = []

    def get_list_with_field_probing(self, request_type, fields, filter=None):
        self.calls.append(request_type)
        if request_type == "GetVolleyTournamentList":
            return self.tournaments, []
        if request_type == "GetVolleyMatchList":
            tno = filter.split("'")[1]
            return self.matches_by_tournament.get(tno, []), []
        raise AssertionError(f"request_type inesperado: {request_type}")


# ---------------------------------------------------------------------------
# is_in_scope
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("FIVB Volleyball Nations League 2025 - Men", True),
        ("World Championship 2026 Women", True),
        ("Paris 2024 Olympic Games Men's Volleyball", True),
        ("CEV EuroVolley 2025 Women", True),
        ("Asian Men's Championship 2023", True),  # bug real: género insertado en medio
        ("African Women's Championship 2022", True),
        ("South American Men's Championship 2021", True),
        ("Club World Championship 2025", False),  # clubes excluidos
        ("U21 World Championship", False),  # juveniles excluidos
        ("Beach Volleyball World Tour", False),  # beach excluido
        ("2005 FIVB Men's World Grand Champions Cup", False),  # no es el World Cup oficial
        ("31st World Military Volleyball Championship-Women", False),  # CISM, no selecciones oficiales
        ("33rd CISM World Military Men's Volleyball Championship", False),
        ("Asian Men's U16 Volleyball Championship Thailand 2025", False),  # youth, faltaba u16
        ("Asian Eastern Zonal Men's Volleyball Championship 2026", False),  # sub-región, no continental completo
        ("Central Asian Volleyball Championship 2022", False),  # sub-región
        ("Some Random Local Cup", False),
    ],
)
def test_is_in_scope(name, expected):
    assert is_in_scope(name) is expected


# ---------------------------------------------------------------------------
# crawler con cliente simulado
# ---------------------------------------------------------------------------
def test_crawler_discovers_and_writes_only_in_scope_tournaments(tmp_path):
    tournaments = [
        {"No": "1001", "Name": "Volleyball Nations League 2025 - Men", "Gender": "men"},
        {"No": "1002", "Name": "Club World Championship 2025", "Gender": "men"},
    ]
    matches = {"1001": [{"No": "5001", "TeamAName": "USA", "TeamBName": "Brazil"}]}

    fake_client = FakeVisClient(tournaments, matches)
    crawler = FivbVisCrawler(output_dir=tmp_path, client=fake_client)

    crawler.crawl()

    written = list(tmp_path.glob("tournament_*.jsonl"))
    assert len(written) == 1
    assert written[0].name == "tournament_1001.jsonl"

    lines = [json.loads(l) for l in written[0].read_text(encoding="utf-8").splitlines()]
    assert lines[0]["_type"] == "tournament"
    assert lines[1]["_type"] == "match"
    assert lines[1]["No"] == "5001"


def test_crawler_resume_skips_already_completed_tournaments(tmp_path):
    tournaments = [{"No": "1001", "Name": "Volleyball Nations League 2025 - Men", "Gender": "men"}]
    fake_client = FakeVisClient(tournaments, {"1001": [{"No": "5001", "TeamAName": "USA", "TeamBName": "Brazil"}]})

    crawler = FivbVisCrawler(output_dir=tmp_path, client=fake_client)
    crawler.crawl()
    assert fake_client.calls.count("GetVolleyMatchList") == 1

    # Segunda pasada: mismo torneo ya completado -> no debe volver a pedir sus partidos.
    fake_client_2 = FakeVisClient(tournaments, {"1001": [{"No": "5001", "TeamAName": "USA", "TeamBName": "Brazil"}]})
    crawler_2 = FivbVisCrawler(output_dir=tmp_path, client=fake_client_2)
    crawler_2.crawl()
    assert fake_client_2.calls.count("GetVolleyMatchList") == 0


def test_crawl_state_roundtrip(tmp_path):
    state_path = tmp_path / "state.json"
    state = CrawlState(completed_tournament_ids={"1", "2", "3"})
    state.save(state_path)

    loaded = CrawlState.load(state_path)
    assert loaded.completed_tournament_ids == {"1", "2", "3"}


# ---------------------------------------------------------------------------
# conversión a RawMatchRow
# ---------------------------------------------------------------------------
def test_canonical_competition_name():
    assert canonical_competition_name("FIVB Volleyball Nations League 2025 - Men") == "Volleyball Nations League"
    assert canonical_competition_name("CEV EuroVolley 2025") == "EuroVolley"
    assert canonical_competition_name("Asian Men's Championship 2023") == "Asian Championship"
    assert canonical_competition_name("African Women's Championship 2022") == "African Championship"
    assert canonical_competition_name("Some Unrelated Event") is None


def test_resolve_gender_from_field_and_fallback():
    assert resolve_gender({"Gender": "men"}, "any name") == "men"
    assert resolve_gender({"Gender": "1"}, "any name") == "women"
    assert resolve_gender({}, "World Championship Women") == "women"
    assert resolve_gender({}, "ambiguous tournament name") is None


def test_extract_date_only_normal_case():
    assert _extract_date_only("2026-06-10T18:00:00") == "2026-06-10"


def test_extract_date_only_fixes_thai_buddhist_era():
    """Caso real: un torneo en Tailandia devolvió '2568-11-08' en vez de '2025-11-08'."""
    assert _extract_date_only("2568-11-08T10:00:00") == "2025-11-08"


def test_extract_date_only_leaves_other_implausible_years_unfixed():
    # 3000 - 543 = 2457, fuera de rango plausible -> no se "arregla" a ciegas,
    # se deja tal cual para que _parse_date lo rechace explícitamente aguas abajo.
    assert _extract_date_only("3000-01-01T00:00:00") == "3000-01-01"


def _write_fixture_tournament(tmp_path: Path, tournament: Dict, matches: List[Dict]) -> Path:
    path = tmp_path / f"tournament_{tournament['No']}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_type": "tournament", **tournament}) + "\n")
        for m in matches:
            fh.write(json.dumps({"_type": "match", **m}) + "\n")
    return path


def test_iter_fivb_vis_raw_matches_happy_path(tmp_path):
    tournament = {"No": "1001", "Name": "Volleyball Nations League 2025 - Men", "Gender": "men"}
    matches = [
        {
            "No": "5001",
            "TeamAName": "USA",
            "TeamBName": "Brazil",
            "DateTimeLocal": "2025-06-11T18:00:00",
            "MatchPointsA": "3",
            "MatchPointsB": "1",
            "PointsTeamASet1": "25", "PointsTeamBSet1": "20",
            "PointsTeamASet2": "25", "PointsTeamBSet2": "22",
            "PointsTeamASet3": "20", "PointsTeamBSet3": "25",
            "PointsTeamASet4": "25", "PointsTeamBSet4": "18",
            "Hall": "Ottawa Arena",
        }
    ]
    _write_fixture_tournament(tmp_path, tournament, matches)

    results = list(iter_fivb_vis_raw_matches(tmp_path))
    assert len(results) == 1
    raw, sets, reason = results[0]

    assert reason is None
    assert raw.source == "fivb_vis"
    assert raw.gender == "men"
    assert raw.competition_name == "Volleyball Nations League"
    assert raw.sets_a == 3 and raw.sets_b == 1
    assert raw.team_a_name == "USA"
    assert len(sets) == 4  # solo 4 sets jugados
    assert sets[0].team_a_points == 25


def test_iter_fivb_vis_raw_matches_rejects_missing_data(tmp_path):
    tournament = {"No": "1001", "Name": "Volleyball Nations League 2025 - Men", "Gender": "men"}
    matches = [{"No": "5002", "TeamAName": "USA"}]  # falta TeamBName, fecha, marcador
    _write_fixture_tournament(tmp_path, tournament, matches)

    results = list(iter_fivb_vis_raw_matches(tmp_path))
    assert len(results) == 1
    raw, sets, reason = results[0]
    assert raw is None
    assert "DateTimeLocal" in reason or "equipo" in reason.lower()
