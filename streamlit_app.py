"""
Dashboard — Fase 10.

Pensado para desplegar GRATIS en Streamlit Community Cloud
(share.streamlit.io), directo desde GitHub. Por eso:

- Importa la lógica de predicción directamente (`src.api.predictor`), sin
  pasar por HTTP — así no hace falta tener la API de FastAPI corriendo
  aparte, algo incómodo en un contenedor gratuito que solo ejecuta un
  proceso.
- `@st.cache_resource` carga los modelos UNA VEZ por sesión de servidor,
  no en cada clic — cargar un CatBoost desde disco no es gratis.
- Los modelos (`models/<gender>/*.cbm`, `*.json`) DEBEN estar commiteados
  en el repo de GitHub — en un despliegue efímero no hay forma de que la
  app "encuentre" archivos que solo existen en tu ordenador.

Uso local:
    streamlit run streamlit_app.py

Desplegar gratis:
    1. Sube este repo a GitHub (incluyendo la carpeta models/ con los
       modelos ya entrenados — confirma que no está en .gitignore).
    2. Entra en https://share.streamlit.io, conecta tu cuenta de GitHub.
    3. "New app" -> elige el repo, rama, y como archivo principal:
       streamlit_app.py
    4. Deploy. En un par de minutos tienes una URL pública gratis.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.api.predictor import GenderModelBundle, predict_match
from src.normalization.competitions import CompetitionCatalog

MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Volley Predictor", page_icon="🏐", layout="centered")


@st.cache_resource(show_spinner="Cargando modelos...")
def load_bundle(gender: str) -> GenderModelBundle | None:
    try:
        return GenderModelBundle.load(MODELS_DIR, gender)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_competition_catalog() -> CompetitionCatalog:
    return CompetitionCatalog.from_csv(ROOT / "configs" / "competitions_seed.csv")


@st.cache_data(show_spinner=False)
def load_competition_options(_catalog: CompetitionCatalog) -> dict[str, str]:
    """{competition_id: nombre legible} — leído directo del CSV semilla, sin depender de un método nuevo."""
    df = pd.read_csv(ROOT / "configs" / "competitions_seed.csv")
    return dict(zip(df["competition_id"], df["name"] + " (" + df["gender"] + ")"))


def _score_chart_df(set_score_probabilities: dict) -> pd.DataFrame:
    order = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]
    return pd.DataFrame(
        {"marcador": order, "probabilidad": [set_score_probabilities[k] for k in order]}
    ).set_index("marcador")


def main() -> None:
    st.title("🏐 Volley Predictor")
    st.caption("Predicción probabilística de partidos entre selecciones nacionales.")

    gender_label = st.radio("Categoría", ["Masculino", "Femenino"], horizontal=True)
    gender = "men" if gender_label == "Masculino" else "women"

    bundle = load_bundle(gender)
    if bundle is None:
        st.error(
            f"No hay modelos entrenados para {gender_label.lower()} en `models/{gender}/`. "
            f"Corre `python scripts/train_final_models.py` y vuelve a desplegar/recargar."
        )
        st.stop()

    catalog = load_competition_catalog()
    competition_options = {
        cid: name for cid, name in load_competition_options(catalog).items() if cid.endswith(f"_{gender.upper()}")
    }

    teams = bundle.known_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Equipo A", teams, index=0)
    with col2:
        remaining = [t for t in teams if t != team_a]
        team_b = st.selectbox("Equipo B", remaining, index=0)

    col3, col4 = st.columns(2)
    with col3:
        match_date = st.date_input("Fecha del partido", value=date.today())
    with col4:
        competition_id = st.selectbox(
            "Competición",
            list(competition_options.keys()),
            format_func=lambda cid: competition_options[cid],
        )

    if st.button("Predecir", type="primary", use_container_width=True):
        with st.spinner("Calculando..."):
            result = predict_match(
                bundle=bundle,
                team_a=team_a,
                team_b=team_b,
                match_date=match_date.strftime("%Y-%m-%d"),
                competition_id=competition_id,
                competition_catalog=catalog,
            )

        st.divider()

        # --- Probabilidad de ganar, la pieza principal ---
        pa, pb = result["p_team_a_wins"], result["p_team_b_wins"]
        c1, c2 = st.columns(2)
        c1.metric(team_a, f"{pa * 100:.1f}%", delta="favorito" if pa > pb else None)
        c2.metric(team_b, f"{pb * 100:.1f}%", delta="favorito" if pb > pa else None)
        st.progress(pa, text=f"{team_a} {pa * 100:.0f}% — {pb * 100:.0f}% {team_b}")

        # --- Marcador de sets ---
        st.subheader("Probabilidad por marcador exacto")
        st.bar_chart(_score_chart_df(result["set_score_probabilities"]))
        st.info(f"Marcador más probable: **{result['most_likely_score']}**")

        # --- Detalles ---
        st.subheader("Detalles")
        d1, d2, d3 = st.columns(3)
        d1.metric("Elo " + team_a, f"{result['elo_team_a']:.0f}")
        d2.metric("Elo " + team_b, f"{result['elo_team_b']:.0f}")
        d3.metric("Dif. de puntos esperada", f"{result['expected_point_diff']:+.1f}")

        st.caption(f"Nivel de confianza de la predicción: **{result['confidence']}**")

        if result["explanatory_factors"]:
            st.subheader("¿Por qué esta predicción?")
            for factor in result["explanatory_factors"]:
                st.markdown(f"- {factor}")
        else:
            st.caption("Sin factores destacables (equipos muy parejos o con poco historial).")


if __name__ == "__main__":
    main()