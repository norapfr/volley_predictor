import numpy as np
import pandas as pd

from src.models.calibration import PlattCalibrator
from src.models.ensemble import make_average_ensemble_factory, make_calibrated_factory


def _constant_factory(value: float):
    def factory(train_df):
        def predict(test_df):
            return np.full(len(test_df), value)

        return predict

    return factory


def test_average_ensemble_simple_mean():
    factories = {"a": _constant_factory(0.2), "b": _constant_factory(0.8)}
    ensemble = make_average_ensemble_factory(factories)

    predict = ensemble(pd.DataFrame({"date": ["2020-01-01"]}))
    test_df = pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-04"]})
    preds = predict(test_df)

    assert np.allclose(preds, 0.5)


def test_average_ensemble_weighted():
    factories = {"a": _constant_factory(0.0), "b": _constant_factory(1.0)}
    ensemble = make_average_ensemble_factory(factories, weights={"a": 0.25, "b": 0.75})

    predict = ensemble(pd.DataFrame({"date": ["2020-01-01"]}))
    test_df = pd.DataFrame({"date": ["2020-01-02", "2020-01-03"]})
    preds = predict(test_df)

    assert np.allclose(preds, 0.75)


def test_calibrated_factory_splits_train_chronologically_not_randomly(synthetic_matches):
    seen_calib_dates = []

    def spy_base_factory(fit_df):
        seen_calib_dates.append(fit_df["date"].max())

        def predict(test_df):
            return np.full(len(test_df), 0.5)

        return predict

    calibrated = make_calibrated_factory(spy_base_factory, PlattCalibrator, calib_fraction=0.2)
    train_df = synthetic_matches.sort_values("date")
    calibrated(train_df)

    # El modelo base solo debe haber visto el primer 80% cronológico, no el train_df completo.
    expected_cutoff = train_df.sort_values("date").iloc[int(len(train_df) * 0.8) - 1]["date"]
    assert seen_calib_dates[0] == expected_cutoff


def test_calibrated_factory_runs_end_to_end(synthetic_matches):
    def base_factory(train_df):
        def predict(test_df):
            return np.full(len(test_df), 0.7)

        return predict

    calibrated = make_calibrated_factory(base_factory, PlattCalibrator)
    train_df = synthetic_matches.iloc[:40]
    test_df = synthetic_matches.iloc[40:]

    predict_fn = calibrated(train_df)
    preds = predict_fn(test_df)

    assert len(preds) == len(test_df)
    assert ((preds >= 0) & (preds <= 1)).all()
