import numpy as np

from src.models.calibration import IsotonicCalibrator, PlattCalibrator


def test_platt_corrects_systematic_overconfidence():
    """Un modelo que siempre dice 90% pero acierta solo el 60% debe calibrarse hacia abajo."""
    rng = np.random.default_rng(0)
    n = 2000
    raw_probs = np.full(n, 0.9)
    y_true = (rng.random(n) < 0.6).astype(float)  # solo acierta el 60% de verdad

    calibrator = PlattCalibrator().fit(raw_probs, y_true)
    calibrated = calibrator.transform(raw_probs)

    assert calibrated.mean() < raw_probs.mean()
    assert abs(calibrated.mean() - 0.6) < 0.05


def test_platt_leaves_already_calibrated_predictions_roughly_unchanged():
    rng = np.random.default_rng(1)
    n = 3000
    raw_probs = rng.uniform(0.1, 0.9, n)
    y_true = (rng.random(n) < raw_probs).astype(float)  # ya perfectamente calibrado por construcción

    calibrator = PlattCalibrator().fit(raw_probs, y_true)
    calibrated = calibrator.transform(raw_probs)

    assert np.corrcoef(raw_probs, calibrated)[0, 1] > 0.9


def test_isotonic_corrects_systematic_overconfidence():
    rng = np.random.default_rng(2)
    n = 2000
    raw_probs = np.full(n, 0.9)
    y_true = (rng.random(n) < 0.6).astype(float)

    calibrator = IsotonicCalibrator().fit(raw_probs, y_true)
    calibrated = calibrator.transform(raw_probs)

    assert abs(calibrated.mean() - 0.6) < 0.05


def test_isotonic_never_outputs_absolute_certainty_with_few_points():
    """
    Con pocos puntos de calibración, la isotónica sin acotar devuelve 0.0/1.0
    exactos (bug real observado con datos reales: log loss se dispara si se
    equivoca en un extremo "imposible"). Debe quedar acotada.
    """
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.3, 0.9, 30)
    y = (rng.random(30) < raw).astype(float)

    calibrator = IsotonicCalibrator().fit(raw, y)
    calibrated = calibrator.transform(raw)

    assert (calibrated > 0.0).all()
    assert (calibrated < 1.0).all()


def test_calibrators_handle_no_variety_without_crashing():
    raw_probs = np.array([0.7, 0.8, 0.6])
    y_true = np.array([1.0, 1.0, 1.0])  # todo positivo, sin variedad

    platt = PlattCalibrator().fit(raw_probs, y_true)
    isotonic = IsotonicCalibrator().fit(raw_probs, y_true)

    # No debe reventar; sin variedad no hay nada que calibrar, se devuelve tal cual.
    assert len(platt.transform(raw_probs)) == 3
    assert len(isotonic.transform(raw_probs)) == 3
