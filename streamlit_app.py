"""
Volley Predictor — Streamlit Dashboard

Modern volleyball prediction interface.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.api.predictor import GenderModelBundle, predict_match
from src.normalization.competitions import CompetitionCatalog


# ============================================================
# CONFIG
# ============================================================

MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Volley Predictor",
    page_icon=str(ROOT / "assets" / "volleyball_remove.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# COUNTRY NAMES
# ============================================================

COUNTRY_NAMES = {
    "ABW": "Aruba",
    "AFG": "Afghanistan",
    "AIA": "Anguilla",
    "ALB": "Albania",
    "ANT": "Netherlands Antilles",
    "ARE": "United Arab Emirates",
    "ARG": "Argentina",
    "ATG": "Antigua and Barbuda",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BDI": "Burundi",
    "BEL": "Belgium",
    "BFA": "Burkina Faso",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BHR": "Bahrain",
    "BHS": "Bahamas",
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BLZ": "Belize",
    "BMU": "Bermuda",
    "BOL": "Bolivia",
    "BRA": "Brazil",
    "BRB": "Barbados",
    "BWA": "Botswana",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "CHN": "China",
    "CMR": "Cameroon",
    "COD": "Democratic Republic of the Congo",
    "COL": "Colombia",
    "CRI": "Costa Rica",
    "CUB": "Cuba",
    "CUW": "Curaçao",
    "CYM": "Cayman Islands",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DMA": "Dominica",
    "DNK": "Denmark",
    "DOM": "Dominican Republic",
    "DZA": "Algeria",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ESP": "Spain",
    "EST": "Estonia",
    "ETH": "Ethiopia",
    "FIN": "Finland",
    "FJI": "Fiji",
    "FRA": "France",
    "FRO": "Faroe Islands",
    "GBR": "United Kingdom",
    "GEO": "Georgia",
    "GHA": "Ghana",
    "GLP": "Guadeloupe",
    "GMB": "Gambia",
    "GRC": "Greece",
    "GRD": "Grenada",
    "GTM": "Guatemala",
    "HKG": "Hong Kong",
    "HND": "Honduras",
    "HRV": "Croatia",
    "HTI": "Haiti",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KGZ": "Kyrgyzstan",
    "KNA": "Saint Kitts and Nevis",
    "KOR": "South Korea",
    "KWT": "Kuwait",
    "LBN": "Lebanon",
    "LBY": "Libya",
    "LCA": "Saint Lucia",
    "LKA": "Sri Lanka",
    "LSO": "Lesotho",
    "LUX": "Luxembourg",
    "LVA": "Latvia",
    "MAC": "Macao",
    "MAR": "Morocco",
    "MDA": "Moldova",
    "MDV": "Maldives",
    "MEX": "Mexico",
    "MKD": "North Macedonia",
    "MLI": "Mali",
    "MNE": "Montenegro",
    "MNG": "Mongolia",
    "MOZ": "Mozambique",
    "MSR": "Montserrat",
    "MTQ": "Martinique",
    "MUS": "Mauritius",
    "NER": "Niger",
    "NGA": "Nigeria",
    "NIC": "Nicaragua",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "OMN": "Oman",
    "PAK": "Pakistan",
    "PAN": "Panama",
    "PER": "Peru",
    "PHL": "Philippines",
    "POL": "Poland",
    "PRI": "Puerto Rico",
    "PRK": "North Korea",
    "PRT": "Portugal",
    "PRY": "Paraguay",
    "PSE": "Palestine",
    "QAT": "Qatar",
    "ROU": "Romania",
    "RUS": "Russia",
    "RWA": "Rwanda",
    "SAU": "Saudi Arabia",
    "SEN": "Senegal",
    "SLV": "El Salvador",
    "SRB": "Serbia",
    "SSD": "South Sudan",
    "SUR": "Suriname",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "TCA": "Turks and Caicos Islands",
    "TCD": "Chad",
    "THA": "Thailand",
    "TKM": "Turkmenistan",
    "TON": "Tonga",
    "TPE": "Taiwan",
    "TTO": "Trinidad and Tobago",
    "TUN": "Tunisia",
    "TUR": "Türkiye",
    "TZA": "Tanzania",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "URY": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
    "VCT": "Saint Vincent and the Grenadines",
    "VEN": "Venezuela",
    "VGB": "British Virgin Islands",
    "VIR": "U.S. Virgin Islands",
    "VNM": "Vietnam",
    "WSM": "Samoa",
    "XKX": "Kosovo",
    "XSCG": "Serbia and Montenegro",
    "ZAF": "South Africa",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
}
def country_name(team: str) -> str:
    return COUNTRY_NAMES.get(team, team)


# ============================================================
# DEFUNCT TEAMS
# ============================================================

# Equipos/uniones disueltas que no deben aparecer como opción
# para partidos futuros, aunque tengan histórico de partidos jugados
# (y por tanto sigan usándose para el Elo/entrenamiento del modelo).
DEFUNCT_TEAMS = {
    "XSCG",  # Serbia y Montenegro (disuelta 2006)
    "XSU",   # URSS
    "XCS",   # Checoslovaquia
    "XYU",   # Yugoslavia
    "ANT",   # Antillas Neerlandesas (disuelta 2010)
}


# ============================================================
# TEAM -> CONTINENT (para filtrar competiciones continentales)
# ============================================================

TEAM_CONTINENT = {
    # EURO (CEV)
    "ALB": "EURO", "AUT": "EURO", "AZE": "EURO", "BEL": "EURO", "BGR": "EURO",
    "BIH": "EURO", "BLR": "EURO", "CHE": "EURO", "CYP": "EURO", "CZE": "EURO",
    "DEU": "EURO", "DNK": "EURO", "ESP": "EURO", "EST": "EURO", "FIN": "EURO",
    "FRA": "EURO", "FRO": "EURO", "GBR": "EURO", "GEO": "EURO", "GRC": "EURO",
    "HRV": "EURO", "HUN": "EURO", "ISL": "EURO", "ISR": "EURO", "ITA": "EURO",
    "LUX": "EURO", "LVA": "EURO", "MDA": "EURO", "MKD": "EURO", "MNE": "EURO",
    "NLD": "EURO", "NOR": "EURO", "POL": "EURO", "PRT": "EURO", "ROU": "EURO",
    "RUS": "EURO", "SRB": "EURO", "SVK": "EURO", "SVN": "EURO", "SWE": "EURO",
    "TUR": "EURO", "UKR": "EURO", "XKX": "EURO",

    # NORCECA
    "ABW": "NORCECA", "AIA": "NORCECA", "ATG": "NORCECA", "BHS": "NORCECA",
    "BLZ": "NORCECA", "BMU": "NORCECA", "BRB": "NORCECA", "CAN": "NORCECA",
    "CRI": "NORCECA", "CUB": "NORCECA", "CUW": "NORCECA", "CYM": "NORCECA",
    "DMA": "NORCECA", "DOM": "NORCECA", "GLP": "NORCECA", "GRD": "NORCECA",
    "GTM": "NORCECA", "HND": "NORCECA", "HTI": "NORCECA", "JAM": "NORCECA",
    "KNA": "NORCECA", "LCA": "NORCECA", "MEX": "NORCECA", "MSR": "NORCECA",
    "MTQ": "NORCECA", "NIC": "NORCECA", "PAN": "NORCECA", "PRI": "NORCECA",
    "SLV": "NORCECA", "TCA": "NORCECA", "TTO": "NORCECA", "USA": "NORCECA",
    "VCT": "NORCECA", "VGB": "NORCECA", "VIR": "NORCECA",

    # SAMER (CSV)
    "ARG": "SAMER", "BOL": "SAMER", "BRA": "SAMER", "CHL": "SAMER",
    "COL": "SAMER", "ECU": "SAMER", "PER": "SAMER", "PRY": "SAMER",
    "SUR": "SAMER", "URY": "SAMER", "VEN": "SAMER",

    # ASIA (AVC)
    "AFG": "ASIA", "ARE": "ASIA", "BGD": "ASIA", "BHR": "ASIA", "CHN": "ASIA",
    "HKG": "ASIA", "IDN": "ASIA", "IND": "ASIA", "IRN": "ASIA", "IRQ": "ASIA",
    "JOR": "ASIA", "JPN": "ASIA", "KAZ": "ASIA", "KGZ": "ASIA", "KOR": "ASIA",
    "KWT": "ASIA", "LBN": "ASIA", "LKA": "ASIA", "MAC": "ASIA", "MDV": "ASIA",
    "MNG": "ASIA", "OMN": "ASIA", "PAK": "ASIA", "PHL": "ASIA", "PRK": "ASIA",
    "PSE": "ASIA", "QAT": "ASIA", "SAU": "ASIA", "TPE": "ASIA", "TKM": "ASIA",
    "THA": "ASIA", "UZB": "ASIA", "VNM": "ASIA",

    # AFRICA (CAVB)
    "BDI": "AFRICA", "BFA": "AFRICA", "BWA": "AFRICA", "CMR": "AFRICA",
    "COD": "AFRICA", "DZA": "AFRICA", "EGY": "AFRICA", "ETH": "AFRICA",
    "GHA": "AFRICA", "GMB": "AFRICA", "KEN": "AFRICA", "LBY": "AFRICA",
    "LSO": "AFRICA", "MAR": "AFRICA", "MLI": "AFRICA", "MOZ": "AFRICA",
    "MUS": "AFRICA", "NER": "AFRICA", "NGA": "AFRICA", "RWA": "AFRICA",
    "SEN": "AFRICA", "SSD": "AFRICA", "TCD": "AFRICA", "TUN": "AFRICA",
    "TZA": "AFRICA", "UGA": "AFRICA", "ZAF": "AFRICA", "ZMB": "AFRICA",
    "ZWE": "AFRICA",

    # Oceanía — sin competición continental en el CSV, se quedan sin match
    "AUS": "OCEANIA", "FJI": "OCEANIA", "NZL": "OCEANIA", "TON": "OCEANIA",
    "WSM": "OCEANIA",
}

CONTINENTAL_PREFIXES = {"EURO", "NORCECA", "SAMER", "ASIA", "AFRICA"}


def is_competition_allowed(competition_id: str, team_a: str, team_b: str) -> bool:
    """Las competiciones abiertas (VNL, Mundial, JJOO, Copa del Mundo)
    siempre valen. Las continentales solo si ambos equipos pertenecen
    a ese continente."""
    prefix = competition_id.rsplit("_", 1)[0]
    if prefix not in CONTINENTAL_PREFIXES:
        return True
    return (
        TEAM_CONTINENT.get(team_a) == prefix
        and TEAM_CONTINENT.get(team_b) == prefix
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   COLOR PALETTE
============================================================

   Background:  #080c11
   Panel:       #111820
   Border:      #29333f
   Orange:      #ff6b35
   Orange 2:    #ff8a5b
   Text:        #ffffff
   Muted:       #8d98a7

============================================================ */


/* ============================================================
   GLOBAL APP
============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 50% -15%,
            rgba(255, 107, 53, 0.18),
            transparent 34%
        ),
        radial-gradient(
            circle at 0% 45%,
            rgba(255, 107, 53, 0.05),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #080c11 0%,
            #0b1016 50%,
            #080c11 100%
        );
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT UI
============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ============================================================
   MAIN TITLE
============================================================ */

h1 {
    text-align: center !important;

    color: #ffffff !important;

    font-size: 48px !important;
    font-weight: 950 !important;

    letter-spacing: -2.5px !important;

    margin-top: 0 !important;
    margin-bottom: 2px !important;

    text-shadow:
        0 0 25px rgba(255, 107, 53, 0.12);
}


/* ============================================================
   SUBTITLE
============================================================ */

div[data-testid="stCaptionContainer"] {
    text-align: center;
}

div[data-testid="stCaptionContainer"] p {
    color: #8d98a7 !important;

    font-size: 15px !important;
    font-weight: 500 !important;

    letter-spacing: 0.3px;
}


/* ============================================================
   SECTION TITLES
============================================================ */

h2 {
    color: #ffffff !important;

    font-size: 27px !important;
    font-weight: 900 !important;

    letter-spacing: -0.5px !important;

    margin-top: 15px !important;
    margin-bottom: 18px !important;

    position: relative;
}


/* Orange accent for h2 */

h2::first-letter {
    color: #ff6b35 !important;
}


h3 {
    color: #ffffff !important;

    font-size: 20px !important;
    font-weight: 850 !important;

    letter-spacing: -0.2px !important;
}


/* ============================================================
   SECTION TITLES — ORANGE ACCENT
============================================================ */

/*
   Adds an orange line before the section headings.
*/

div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stHeadingWithActionElements"] h2 {
    border-left: 4px solid #ff6b35;

    padding-left: 12px;
}


/* ============================================================
   LABELS
============================================================ */

label {
    color: #aeb7c4 !important;

    font-weight: 700 !important;
}


/* ============================================================
   SELECTBOX
============================================================ */

div[data-baseweb="select"] > div {
    background-color: #111820 !important;

    border: 1px solid #29333f !important;
    border-radius: 13px !important;

    min-height: 48px !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: #ff6b35 !important;

    box-shadow:
        0 0 0 1px rgba(255, 107, 53, 0.15);
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}


/* Dropdown */

div[role="listbox"] {
    background-color: #111820 !important;

    border: 1px solid #29333f !important;
}

div[role="option"] {
    color: #ffffff !important;
}

div[role="option"]:hover {
    background-color: rgba(255, 107, 53, 0.12) !important;
}


/* ============================================================
   DATE INPUT
============================================================ */

div[data-baseweb="input"] > div {
    background-color: #111820 !important;

    border: 1px solid #29333f !important;
    border-radius: 13px !important;

    min-height: 48px !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

div[data-baseweb="input"] > div:hover {
    border-color: #ff6b35 !important;

    box-shadow:
        0 0 0 1px rgba(255, 107, 53, 0.15);
}

input {
    color: #ffffff !important;
}


/* ============================================================
   RADIO BUTTONS
============================================================ */

div[role="radiogroup"] {
    gap: 10px;
}

div[role="radiogroup"] label {
    background-color: #111820;

    border: 1px solid #29333f;

    border-radius: 11px;

    padding: 8px 20px;

    transition: all 0.2s ease;
}

div[role="radiogroup"] label:hover {
    border-color: #ff6b35;

    background-color: rgba(255, 107, 53, 0.06);
}


/* ============================================================
   VS
============================================================ */

/* VS — perfectamente centrado entre Team A y Team B */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) h3 {
    text-align: center !important;
    color: #ff6b35 !important;
    font-size: 24px !important;
    font-weight: 900 !important;
    letter-spacing: 3px !important;
    line-height: 1 !important;

    margin: 0 !important;
    padding: 0 !important;

    text-shadow: 0 0 18px rgba(255, 107, 53, 0.35);
}


/* ============================================================
   PRIMARY BUTTON
============================================================ */

.stButton > button {
    width: 100%;

    height: 60px;

    border-radius: 15px;

    border: 1px solid #ff6b35;

    background:
        linear-gradient(
            135deg,
            #ff6b35,
            #f4511e
        );

    color: #ffffff;

    font-size: 16px;
    font-weight: 950;

    letter-spacing: 1px;

    box-shadow:
        0 8px 25px rgba(255, 107, 53, 0.18);

    transition:
        all 0.2s ease;
}

.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #ff7b49,
            #ff5b27
        );

    color: #ffffff;

    border-color: #ff8a5b;

    box-shadow:
        0 10px 30px rgba(255, 107, 53, 0.28);

    transform: translateY(-2px);
}

.stButton > button:active {
    transform: translateY(0);
}


/* ============================================================
   METRIC BLOCKS
============================================================ */

div[data-testid="stMetric"] {
    position: relative;

    background:
        linear-gradient(
            145deg,
            #121a22,
            #0f151c
        );

    border: 1px solid #29333f;

    border-radius: 18px;

    padding: 22px;

    min-height: 120px;

    overflow: hidden;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.15);
}


/* Orange top accent */

div[data-testid="stMetric"]::before {
    content: "";

    position: absolute;

    top: 0;
    left: 0;

    width: 100%;
    height: 3px;

    background:
        linear-gradient(
            90deg,
            #ff6b35,
            rgba(255, 107, 53, 0.15)
        );
}


div[data-testid="stMetricLabel"] {
    color: #8f9aa8 !important;

    font-size: 13px !important;
    font-weight: 700 !important;

    text-transform: uppercase;
    letter-spacing: 0.4px;
}


div[data-testid="stMetricValue"] {
    color: #ffffff !important;

    font-size: 34px !important;
    font-weight: 950 !important;
}


div[data-testid="stMetricDelta"] {
    font-weight: 700 !important;
}


/* ============================================================
   PROGRESS BAR
============================================================ */

div[data-testid="stProgressBar"] {
    margin-top: 15px;
    margin-bottom: 20px;
}

div[data-testid="stProgressBar"] > div {
    background-color: #252e38;

    border-radius: 10px;

    height: 10px;
}

div[data-testid="stProgressBar"] > div > div {
    background:
        linear-gradient(
            90deg,
            #ff6b35,
            #ff8a5b
        );

    border-radius: 10px;
}


/* ============================================================
   INFO / RESULT BOXES
============================================================ */

div[data-testid="stAlert"] {
    background:
        linear-gradient(
            135deg,
            #121a22,
            #10161d
        );

    border: 1px solid #29333f;

    border-left: 4px solid #ff6b35;

    border-radius: 14px;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.12);
}


/* ============================================================
   DIVIDERS
============================================================ */

hr {
    border: none !important;

    border-top: 1px solid #242d37 !important;

    margin-top: 32px !important;
    margin-bottom: 32px !important;
}


/* ============================================================
   CHART
============================================================ */

div[data-testid="stVegaLiteChart"] {
    background:
        linear-gradient(
            145deg,
            #121a22,
            #0f151c
        );

    border: 1px solid #29333f;

    border-radius: 18px;

    padding: 10px;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.12);
}


/* ============================================================
   TEXT
============================================================ */

p {
    color: #d6dbe1;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 36px !important;
        letter-spacing: -1.5px !important;
    }

    h2 {
        font-size: 23px !important;
    }

    div[data-testid="stColumn"]:nth-child(2) h3 {
        font-size: 18px !important;
        letter-spacing: 2px !important;

        margin-top: 27px !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading prediction model...")
def load_bundle(
    gender: str,
) -> GenderModelBundle | None:

    try:
        return GenderModelBundle.load(
            MODELS_DIR,
            gender,
        )

    except FileNotFoundError:
        return None


# ============================================================
# LOAD COMPETITION CATALOG
# ============================================================

@st.cache_resource(show_spinner=False)
def load_competition_catalog() -> CompetitionCatalog:

    return CompetitionCatalog.from_csv(
        ROOT / "configs" / "competitions_seed.csv"
    )


# ============================================================
# LOAD COMPETITION OPTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_competition_options(
    _catalog: CompetitionCatalog,
) -> dict[str, str]:

    df = pd.read_csv(
        ROOT / "configs" / "competitions_seed.csv"
    )

    return dict(
        zip(
            df["competition_id"],
            df["name"] + " (" + df["gender"] + ")",
        )
    )


# ============================================================
# SCORE CHART
# ============================================================

def _score_chart_df(
    set_score_probabilities: dict,
) -> pd.DataFrame:

    order = [
        "3-0",
        "3-1",
        "3-2",
        "2-3",
        "1-3",
        "0-3",
    ]

    return pd.DataFrame(
        {
            "Score": order,
            "Probability": [
                set_score_probabilities[k]
                for k in order
            ],
        }
    ).set_index("Score")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "VOLLEY PREDICTOR"
    )

    st.caption(
        "International volleyball match prediction"
    )


    # ========================================================
    # COMPETITION
    # ========================================================

    st.divider()

    st.subheader(
        "Competition"
    )

    gender_label = st.radio(
        "Category",
        ["Men", "Women"],
        horizontal=True,
    )

    gender = (
        "men"
        if gender_label == "Men"
        else "women"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    bundle = load_bundle(gender)

    if bundle is None:

        st.error(
            f"No trained model available for "
            f"{gender_label.lower()}."
        )

        st.info(
            f"Make sure the trained model exists in "
            f"`models/{gender}/`."
        )

        st.stop()


    # ========================================================
    # COMPETITION CATALOG
    # ========================================================

    catalog = load_competition_catalog()

    competition_options = {
        cid: name
        for cid, name
        in load_competition_options(catalog).items()
        if cid.endswith(
            f"_{gender.upper()}"
        )
    }


    # ========================================================
    # TEAMS
    # ========================================================

    teams = bundle.known_teams()

    teams = [
        team
        for team in teams
        if team not in DEFUNCT_TEAMS
    ]

    team_display = {
        team: country_name(team)
        for team in teams
    }

    # Ordenar alfabéticamente por nombre del país
    teams = sorted(
        teams,
        key=lambda team: country_name(team).lower(),
    )


    # ========================================================
    # MATCHUP
    # ========================================================

    st.subheader(
        "Matchup"
    )

    col1, col2, col3 = st.columns(
        [5, 1, 5],
        vertical_alignment="center",
    )


    # --------------------------------------------------------
    # TEAM A
    # --------------------------------------------------------

    with col1:

        team_a = st.selectbox(
            "Team A",
            teams,
            index=0,
            format_func=lambda x: team_display[x],
        )


    # --------------------------------------------------------
    # VS
    # --------------------------------------------------------

    with col2:
        st.markdown("### VS")


    # --------------------------------------------------------
    # TEAM B
    # --------------------------------------------------------

    with col3:

        remaining = [
            team
            for team in teams
            if team != team_a
        ]

        team_b = st.selectbox(
            "Team B",
            remaining,
            index=0,
            format_func=lambda x: team_display[x],
        )


    # ========================================================
    # DATE + COMPETITION
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        match_date = st.date_input(
            "Match date",
            value=date.today(),
        )


    with col2:

        filtered_competitions = {
            cid: name
            for cid, name in competition_options.items()
            if is_competition_allowed(cid, team_a, team_b)
        }

        competition_id = st.selectbox(
            "Competition",
            list(filtered_competitions.keys()),
            format_func=lambda cid:
                filtered_competitions[cid],
        )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    st.write("")

    predict_clicked = st.button(
        "🏐  PREDICT MATCH",
        type="primary",
        use_container_width=True,
    )


    if not predict_clicked:
        return


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    with st.spinner(
        "Analyzing volleyball match..."
    ):

        result = predict_match(
            bundle=bundle,
            team_a=team_a,
            team_b=team_b,
            match_date=match_date.strftime(
                "%Y-%m-%d"
            ),
            competition_id=competition_id,
            competition_catalog=catalog,
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction"
    )

    pa = result["p_team_a_wins"]
    pb = result["p_team_b_wins"]


    # --------------------------------------------------------
    # WIN PROBABILITIES
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            country_name(team_a),
            f"{pa * 100:.1f}%",
            "★ Favorite"
            if pa > pb
            else None,
        )


    with c2:

        st.metric(
            country_name(team_b),
            f"{pb * 100:.1f}%",
            "★ Favorite"
            if pb > pa
            else None,
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    st.progress(
        pa,
        text=(
            f"{country_name(team_a)} "
            f"{pa * 100:.0f}%  —  "
            f"{pb * 100:.0f}% "
            f"{country_name(team_b)}"
        ),
    )


    # ========================================================
    # MOST LIKELY RESULT
    # ========================================================

    st.subheader(
        "Most likely result"
    )

    score_probability = result.get(
        "most_likely_score_probability",
        result["set_score_probabilities"][result["most_likely_score"]],
    )
    score_confidence = result.get(
        "score_confidence",
        "low",
    )
    confidence_label = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }.get(score_confidence, str(score_confidence).title())

    st.info(
        f"🏐 Most likely score: "
        f"**{result['most_likely_score']}** "
        f"({score_probability * 100:.1f}%)"
    )

    a1, a2 = st.columns(2)

    with a1:

        st.metric(
            "Exact score confidence",
            confidence_label,
        )

    with a2:

        st.metric(
            "Lead over next score",
            f"{result.get('score_confidence_gap', 0.0) * 100:.1f} pp",
        )

    close_alternatives = result.get(
        "close_score_alternatives",
        [],
    )

    if close_alternatives:

        alternatives_text = ", ".join(
            f"{item['score']} ({item['probability'] * 100:.1f}%)"
            for item in close_alternatives
        )

        st.caption(
            f"Close alternatives: {alternatives_text}"
        )

    if score_confidence == "low":

        st.warning(
            "Exact score is not very decisive; check the nearby alternatives before treating it as a strong call."
        )


    # --------------------------------------------------------
    # SCORE PROBABILITIES
    # --------------------------------------------------------

    st.bar_chart(
        _score_chart_df(
            result["set_score_probabilities"]
        ),
        color="#c25932",
    )


    # ========================================================
    # MATCH STATISTICS
    # ========================================================

    st.subheader(
        "Match statistics"
    )

    d1, d2, d3 = st.columns(3)


    with d1:

        st.metric(
            f"Elo — {country_name(team_a)}",
            f"{result['elo_team_a']:.0f}",
        )


    with d2:

        st.metric(
            f"Elo — {country_name(team_b)}",
            f"{result['elo_team_b']:.0f}",
        )


    with d3:

        st.metric(
            "Expected point difference",
            f"{result['expected_point_diff']:+.1f}",
        )


    # ========================================================
    # MODEL ASSESSMENT
    # ========================================================

    st.subheader(
        "Model assessment"
    )

    st.info(
        f"Prediction confidence: "
        f"**{result['confidence']}**"
    )


    # ========================================================
    # EXPLANATORY FACTORS
    # ========================================================

    st.subheader(
        "Why this prediction?"
    )

    factors = result.get(
        "explanatory_factors",
        [],
    )

    if factors:

        for factor in factors:

            st.markdown(
                f"• {factor}"
            )

    else:

        st.caption(
            "No significant factors detected "
            "for this matchup."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
