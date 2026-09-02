from pathlib import Path

import pandas as pd
import pytest

from src.normalization.competitions import CompetitionCatalog
from src.normalization.teams import TeamNormalizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def team_normalizer() -> TeamNormalizer:
    return TeamNormalizer.from_csv(ROOT / "configs" / "team_aliases_seed.csv")


@pytest.fixture
def competition_catalog() -> CompetitionCatalog:
    return CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_vnl_men.csv"


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """
    60 partidos sintéticos, hombres, entre 4 equipos con una jerarquía de
    fuerza clara y estable: STRONG > MID > WEAK > BOTTOM casi siempre.
    Alterna qué equipo va en la posición team_a/team_b entre rondas, para
    que el resultado no esté confundido con la posición (si el más fuerte
    fuera siempre "team_a", cualquier modelo tonto acertaría sin aprender
    nada real sobre la fuerza de cada equipo).
    """
    rows = []
    order = ["STRONG", "MID", "WEAK", "BOTTOM"]  # de más a menos fuerte
    rank = {team: i for i, team in enumerate(order)}
    match_id = 0
    date = pd.Timestamp("2015-01-01")
    for round_ in range(10):
        for i, team_x in enumerate(order):
            for team_y in order[i + 1:]:
                # Alterna la posición para no confundir "team_a" con "más fuerte".
                if round_ % 2 == 0:
                    team_a, team_b = team_x, team_y
                else:
                    team_a, team_b = team_y, team_x

                stronger = team_a if rank[team_a] < rank[team_b] else team_b
                upset = round_ == 7 and {team_a, team_b} == {"BOTTOM", "STRONG"}
                if upset:
                    winner = "team_b" if stronger == team_a else "team_a"
                else:
                    winner = "team_a" if stronger == team_a else "team_b"

                # Marcador de sets sintético coherente con el ganador (3-0/3-1/3-2 al azar determinista).
                margin = [(3, 0), (3, 1), (3, 2)][match_id % 3]
                if winner == "team_a":
                    sets_a, sets_b = margin
                else:
                    sets_b, sets_a = margin

                rows.append(
                    {
                        "match_id": f"m{match_id}",
                        "date": (date + pd.Timedelta(days=match_id)).strftime("%Y-%m-%d"),
                        "gender": "men",
                        "team_a": team_a,
                        "team_b": team_b,
                        "sets_a": sets_a,
                        "sets_b": sets_b,
                        "winner": winner,
                    }
                )
                match_id += 1
    return pd.DataFrame(rows)
