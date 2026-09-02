import numpy as np

from src.evaluation.metrics import accuracy, brier_score, log_loss, summarize


def test_log_loss_perfect_predictions_near_zero():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.999, 0.001, 0.999, 0.001])
    assert log_loss(y_true, y_pred) < 0.01


def test_log_loss_confidently_wrong_is_heavily_penalized():
    y_true = np.array([1.0])
    y_pred_wrong_confident = np.array([0.01])
    y_pred_right_unsure = np.array([0.6])
    assert log_loss(y_true, y_pred_wrong_confident) > log_loss(y_true, y_pred_right_unsure)


def test_brier_score_range():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([1.0, 0.0])
    assert brier_score(y_true, y_pred) == 0.0

    y_pred_worst = np.array([0.0, 1.0])
    assert brier_score(y_true, y_pred_worst) == 1.0


def test_accuracy_basic():
    y_true = np.array([1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.9, 0.1, 0.4, 0.6])  # el tercero se equivoca (predice <0.5 pero y_true=1)
    assert accuracy(y_true, y_pred) == 0.75


def test_summarize_has_all_keys():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.7, 0.3])
    result = summarize(y_true, y_pred)
    assert set(result.keys()) == {"log_loss", "brier_score", "accuracy", "n"}
    assert result["n"] == 2
