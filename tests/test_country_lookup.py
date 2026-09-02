from src.normalization.country_lookup import resolve_country
from src.normalization.teams import TeamNormalizer


def test_resolve_country_standard_names():
    assert resolve_country("Brazil") == "BRA"
    assert resolve_country("Netherlands") == "NLD"
    assert resolve_country("South Korea") == "KOR"


def test_resolve_country_fivb_specific_overrides():
    assert resolve_country("Chinese Taipei") == "TPE"
    assert resolve_country("Ivory Coast") == "CIV"
    assert resolve_country("DR Congo") == "COD"
    assert resolve_country("USA") == "USA"


def test_resolve_country_historical_teams():
    assert resolve_country("Soviet Union") == "XSU"
    assert resolve_country("Czechoslovakia") == "XCS"
    assert resolve_country("Yugoslavia") == "XYU"


def test_resolve_country_unknown_returns_none():
    assert resolve_country("Atlantis") is None
    assert resolve_country("Not A Real Country") is None


def test_resolve_country_real_edge_cases_from_vis_crawl():
    """Casos reales encontrados al crawlear FIVB VIS que no resolvía la versión anterior."""
    assert resolve_country("Germany (RED)") == "DEU"  # sufijo de escuadra entre paréntesis
    assert resolve_country("Japan (WHITE)") == "JPN"
    assert resolve_country("Maldive Islands") == "MDV"
    assert resolve_country("Moldovia") == "MDA"  # errata de VIS por Moldova
    assert resolve_country("Netherlands Antilles") == "ANT"
    assert resolve_country("Samoa, Western") == "WSM"
    assert resolve_country("Turkey") == "TUR"  # ISO renombró a "Türkiye"; VIS sigue con "Turkey"
    assert resolve_country("Turkez") == "TUR"  # errata de VIS
    assert resolve_country("U.S.A.") == "USA"
    assert resolve_country("China, People's Rep. of") == "CHN"
    assert resolve_country("Democratic Republic of Congo") == "COD"
    assert resolve_country("Hongkong") == "HKG"
    assert resolve_country("Kuweit") == "KWT"  # errata de VIS por Kuwait
    assert resolve_country("Macao, China") == "MAC"
    assert resolve_country("Mali Republic") == "MLI"
    assert resolve_country("Nertherlands") == "NLD"  # errata de VIS por Netherlands
    assert resolve_country("Saudia Arabia") == "SAU"  # errata de VIS por Saudi Arabia


def test_team_normalizer_falls_back_to_country_lookup():
    normalizer = TeamNormalizer()  # sin CSV cargado
    assert normalizer.resolve("fivb_vis", "Cuba") == "CUB"
    assert normalizer.resolve("fivb_vis", "Poland") == "POL"
    assert normalizer.resolve("fivb_vis", "Chinese Taipei") == "TPE"

    auto = normalizer.auto_resolved_names()
    assert ("fivb_vis", "Cuba", "CUB") in auto


def test_team_normalizer_csv_alias_takes_priority_over_auto():
    normalizer = TeamNormalizer()
    normalizer.add_alias(canonical_team_id="XXX", source="kaggle", source_name="USA (US)")
    assert normalizer.resolve("kaggle", "USA (US)") == "XXX"  # el alias manual gana, no ISO real


def test_team_normalizer_still_reports_truly_unknown():
    normalizer = TeamNormalizer()
    assert normalizer.resolve("fivb_vis", "Atlantis") is None
    assert ("fivb_vis", "Atlantis") in normalizer.unresolved_names()
