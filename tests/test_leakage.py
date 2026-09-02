"""
Test del test crítico de la spec (secciones 6 y 28):

    feature(match_T) no puede depender de result(match_T)

Este test valida la UTILIDAD de detección (`assert_no_future_leakage`) con
casos de juguete. Cuando exista feature engineering real (Fase 3), cada
tabla de features debe pasar por esta misma función antes de aceptarse.
"""

from datetime import date

from src.evaluation.leakage import FeatureProvenance, assert_no_future_leakage


def test_clean_features_produce_no_violations():
    rows = [
        FeatureProvenance(
            match_id="m3",
            match_date=date(2025, 6, 14),
            source_match_dates=[date(2025, 6, 11), date(2025, 6, 12)],
        )
    ]
    assert assert_no_future_leakage(rows) == []


def test_feature_using_same_day_result_is_flagged():
    rows = [
        FeatureProvenance(
            match_id="m3",
            match_date=date(2025, 6, 14),
            source_match_dates=[date(2025, 6, 11), date(2025, 6, 14)],  # incluye el propio partido
        )
    ]
    violations = assert_no_future_leakage(rows)
    assert len(violations) == 1
    assert violations[0].match_id == "m3"


def test_feature_using_future_match_is_flagged():
    rows = [
        FeatureProvenance(
            match_id="m1",
            match_date=date(2025, 6, 11),
            source_match_dates=[date(2025, 6, 14)],  # partido futuro
        )
    ]
    violations = assert_no_future_leakage(rows)
    assert len(violations) == 1
