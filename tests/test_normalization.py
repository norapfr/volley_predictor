def test_resolves_known_alias(team_normalizer):
    assert team_normalizer.resolve("kaggle_vnl", "United States") == "USA"
    assert team_normalizer.resolve("kaggle_vnl", "Estados Unidos") == "USA"
    assert team_normalizer.resolve("fivb_vis", "USA") == "USA"


def test_resolution_is_case_and_accent_insensitive(team_normalizer):
    assert team_normalizer.resolve("kaggle_vnl", "brasil") == "BRA"
    assert team_normalizer.resolve("kaggle_vnl", "BRASIL") == "BRA"
    assert team_normalizer.resolve("kaggle_vnl", "  Brasil  ") == "BRA"


def test_unknown_team_returns_none_and_is_tracked(team_normalizer):
    result = team_normalizer.resolve("kaggle_vnl", "Atlantis")
    assert result is None
    assert ("kaggle_vnl", "Atlantis") in team_normalizer.unresolved_names()


def test_competition_catalog_resolves_by_name_and_gender(competition_catalog):
    comp = competition_catalog.resolve("Volleyball Nations League", "men")
    assert comp is not None
    assert comp.competition_id == "VNL_MEN"

    women_comp = competition_catalog.resolve("volleyball nations league", "women")
    assert women_comp is not None
    assert women_comp.competition_id == "VNL_WOMEN"


def test_unknown_competition_resolves_to_none(competition_catalog):
    assert competition_catalog.resolve("Beach Volleyball World Tour", "men") is None
