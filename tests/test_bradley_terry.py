from src.models.bradley_terry import BradleyTerryModel


def test_bradley_terry_learns_strength_hierarchy(synthetic_matches):
    model = BradleyTerryModel(C=1.0)
    model.fit(synthetic_matches)
    s = model.strengths_
    assert s["STRONG"] > s["MID"] > s["WEAK"] > s["BOTTOM"]


def test_bradley_terry_predicts_stronger_team_favored(synthetic_matches):
    model = BradleyTerryModel(C=1.0)
    model.fit(synthetic_matches)
    assert model.predict_proba("STRONG", "BOTTOM") > 0.5
    assert model.predict_proba("BOTTOM", "STRONG") < 0.5


def test_bradley_terry_symmetric():
    import pandas as pd

    df = pd.DataFrame(
        [{"date": "2020-01-01", "team_a": "A", "team_b": "B", "winner": "team_a"}] * 5
        + [{"date": "2020-01-02", "team_a": "B", "team_a_2": None, "team_b": "A", "winner": "team_b"}] * 5
    )
    model = BradleyTerryModel(C=1.0)
    model.fit(df.drop(columns=["team_a_2"], errors="ignore"))
    p_ab = model.predict_proba("A", "B")
    p_ba = model.predict_proba("B", "A")
    assert abs(p_ab - (1 - p_ba)) < 1e-9


def test_bradley_terry_unseen_team_gets_neutral_strength(synthetic_matches):
    model = BradleyTerryModel(C=1.0)
    model.fit(synthetic_matches)
    p = model.predict_proba("STRONG", "NEVER_SEEN_TEAM")
    assert 0.5 < p < 1.0  # STRONG sigue favorito, pero sin la confianza extrema de un equipo conocido y débil


def test_bradley_terry_no_variety_does_not_crash():
    import pandas as pd

    df = pd.DataFrame(
        [{"date": "2020-01-01", "team_a": "A", "team_b": "B", "winner": "team_a"}] * 5
    )
    model = BradleyTerryModel(C=1.0)
    model.fit(df)  # todos los resultados son "team_a gana" -> sin variedad de clase
    p = model.predict_proba("A", "B")
    assert 0.0 <= p <= 1.0
