import json

import pytest


@pytest.fixture
def trained_models_dir(tmp_path, synthetic_matches, competition_catalog):
    """Entrena y guarda un set de modelos de verdad (con datos sintéticos) para probar la API contra algo real."""
    pytest.importorskip("catboost")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_final_models import train_for_gender

    df = synthetic_matches.copy()
    df["competition_id"] = "VNL_MEN"
    df["stage"] = "pool"
    df["team_a_score"] = 75 + (df["sets_a"] - df["sets_b"]) * 5
    df["team_b_score"] = 75 - (df["sets_a"] - df["sets_b"]) * 5

    out_dir = tmp_path / "models" / "men"
    train_for_gender(df, "men", competition_catalog, out_dir)
    return tmp_path / "models"


def test_gender_model_bundle_loads(trained_models_dir):
    from src.api.predictor import GenderModelBundle

    bundle = GenderModelBundle.load(trained_models_dir, "men")
    assert bundle.gender == "men"
    assert len(bundle.known_teams()) == 4  # STRONG/MID/WEAK/BOTTOM del fixture


def test_gender_model_bundle_missing_raises_clear_error(tmp_path):
    from src.api.predictor import GenderModelBundle

    with pytest.raises(FileNotFoundError):
        GenderModelBundle.load(tmp_path, "women")


def test_predict_match_end_to_end(trained_models_dir, competition_catalog):
    from src.api.predictor import GenderModelBundle, predict_match

    bundle = GenderModelBundle.load(trained_models_dir, "men")
    result = predict_match(
        bundle=bundle,
        team_a="STRONG",
        team_b="BOTTOM",
        match_date="2026-01-01",
        competition_id="VNL_MEN",
        competition_catalog=competition_catalog,
    )

    assert result["p_team_a_wins"] + result["p_team_b_wins"] == pytest.approx(1.0)
    assert result["most_likely_score"] in {"3-0", "3-1", "3-2", "2-3", "1-3", "0-3"}
    assert sum(result["set_score_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["confidence"] in {"alta", "media", "baja"}
    assert isinstance(result["explanatory_factors"], list)
    # STRONG es objetivamente más fuerte en el fixture -> debería ser favorito.
    assert result["p_team_a_wins"] > 0.5


def test_predict_match_unknown_team_does_not_crash(trained_models_dir, competition_catalog):
    from src.api.predictor import GenderModelBundle, predict_match

    bundle = GenderModelBundle.load(trained_models_dir, "men")
    result = predict_match(
        bundle=bundle,
        team_a="NEVER_SEEN",
        team_b="STRONG",
        match_date="2026-01-01",
        competition_id="VNL_MEN",
        competition_catalog=competition_catalog,
    )
    assert 0.0 <= result["p_team_a_wins"] <= 1.0
    assert result["confidence"].startswith("baja")

# ---------------------------------------------------------------------------
# Endpoints HTTP (TestClient — no necesita levantar un servidor real)
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client(trained_models_dir, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import src.api.app as app_module

    monkeypatch.setattr(app_module, "MODELS_DIR", trained_models_dir)
    app_module._bundles.clear()
    return TestClient(app_module.app)


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_teams_endpoint(api_client):
    response = api_client.get("/teams/men")
    assert response.status_code == 200
    body = response.json()
    assert body["gender"] == "men"
    assert body["n_teams"] == 4


def test_teams_endpoint_invalid_gender(api_client):
    response = api_client.get("/teams/robots")
    assert response.status_code == 422


def test_predict_endpoint(api_client):
    response = api_client.post(
        "/predict",
        json={
            "gender": "men",
            "team_a": "STRONG",
            "team_b": "WEAK",
            "date": "2026-01-01",
            "competition_id": "VNL_MEN",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["team_a"] == "STRONG"
    assert body["p_team_a_wins"] > body["p_team_b_wins"]


def test_predict_endpoint_same_team_rejected(api_client):
    response = api_client.post(
        "/predict",
        json={
            "gender": "men",
            "team_a": "STRONG",
            "team_b": "STRONG",
            "date": "2026-01-01",
            "competition_id": "VNL_MEN",
        },
    )
    assert response.status_code == 422


def test_predict_endpoint_unknown_competition_rejected(api_client):
    response = api_client.post(
        "/predict",
        json={
            "gender": "men",
            "team_a": "STRONG",
            "team_b": "WEAK",
            "date": "2026-01-01",
            "competition_id": "DOES_NOT_EXIST",
        },
    )
    assert response.status_code == 422


def test_predict_endpoint_missing_gender_models_returns_503(api_client):
    response = api_client.get("/teams/women")  # no se entrenaron modelos de mujeres en este fixture
    assert response.status_code == 503
