import pandas as pd
import pytest

from src.features.competition_importance import add_importance_features
from src.normalization.competitions import CompetitionCatalog

pytestmark = pytest.mark.usefixtures("competition_catalog")


def test_olympic_final_has_max_importance(competition_catalog: CompetitionCatalog):
    df = pd.DataFrame([{"competition_id": "OG_MEN", "stage": "final"}])
    result = add_importance_features(df, competition_catalog)
    assert result.iloc[0]["competition_importance"] == 5.0
    assert result.iloc[0]["stage_importance"] == 4.0
    assert result.iloc[0]["match_importance"] == 20.0


def test_continental_pool_stage_has_low_importance(competition_catalog: CompetitionCatalog):
    df = pd.DataFrame([{"competition_id": "NORCECA_MEN", "stage": "pool"}])
    result = add_importance_features(df, competition_catalog)
    assert result.iloc[0]["competition_importance"] == 2.0
    assert result.iloc[0]["stage_importance"] == 1.0


def test_final_more_important_than_pool_same_competition(competition_catalog: CompetitionCatalog):
    df = pd.DataFrame(
        [
            {"competition_id": "VNL_MEN", "stage": "pool"},
            {"competition_id": "VNL_MEN", "stage": "final"},
        ]
    )
    result = add_importance_features(df, competition_catalog)
    assert result.iloc[1]["match_importance"] > result.iloc[0]["match_importance"]


def test_unknown_competition_falls_back_to_default(competition_catalog: CompetitionCatalog):
    df = pd.DataFrame([{"competition_id": "SOME_UNKNOWN_ID", "stage": "final"}])
    result = add_importance_features(df, competition_catalog)
    assert result.iloc[0]["competition_importance"] == 1.0  # default, no crashea
