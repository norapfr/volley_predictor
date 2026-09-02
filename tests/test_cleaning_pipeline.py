from src.cleaning.matches import build_matches
from src.ingestion.kaggle_vnl import KaggleVNLSource


def test_pipeline_end_to_end(sample_csv_path, team_normalizer, competition_catalog):
    source = KaggleVNLSource(sample_csv_path)
    rows = list(source.iter_matches())

    result = build_matches(rows, team_normalizer, competition_catalog)

    # 6 filas crudas: 1 duplicado exacto + 1 equipo desconocido (Atlantis) -> 4 partidos válidos
    assert len(result.matches) == 4
    assert result.duplicates_skipped == 1
    assert len(result.rejected) == 1
    assert "Atlantis" in result.rejected[0].reason

    match_ids = {m.match_id for m in result.matches}
    assert len(match_ids) == len(result.matches)  # sin duplicados de match_id

    for m in result.matches:
        assert m.team_a != m.team_b
        assert m.gender.value == "men"


def test_unknown_team_is_reported_not_silently_dropped(sample_csv_path, team_normalizer, competition_catalog):
    source = KaggleVNLSource(sample_csv_path)
    rows = list(source.iter_matches())
    result = build_matches(rows, team_normalizer, competition_catalog)

    reasons = [r.reason for r in result.rejected]
    assert any("Equipo desconocido" in r for r in reasons)


def test_implausible_date_is_rejected_not_crashed(team_normalizer, competition_catalog):
    """
    Fecha corrupta real observada en VIS (calendario budista tailandés sin
    corregir, o cualquier otra corrupción de fecha): debe rechazarse con un
    motivo claro, nunca colar una fecha imposible que rompa pandas más
    adelante en el análisis (ver src/cleaning/matches.py::MAX_PLAUSIBLE_DATE).
    """
    from src.ingestion.base import RawMatchRow

    bad_row = RawMatchRow(
        source="fivb_vis",
        source_match_id="999",
        date_str="2568-11-08",  # sin corregir por _extract_date_only, a propósito
        gender="men",
        competition_name="Volleyball Nations League",
        stage_str=None,
        team_a_name="USA",
        team_b_name="Brazil",
        team_a_score=75,
        team_b_score=70,
        sets_a=3,
        sets_b=1,
        venue=None,
        country_host=None,
        neutral=None,
    )
    result = build_matches([bad_row], team_normalizer, competition_catalog)
    assert len(result.matches) == 0
    assert len(result.rejected) == 1
    assert "rango" in result.rejected[0].reason.lower() or "2568" in result.rejected[0].reason
