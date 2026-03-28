"""
FootPredict-Pro — Streamlit Web UI

A beautiful, interactive football prediction dashboard.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import streamlit as st
import numpy as np

from src.inference.predict_match import MatchPredictor
from src.models.ensemble import MatchPrediction

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FootPredict-Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load predictor (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_predictor() -> MatchPredictor:
    predictor = MatchPredictor()
    predictor.load()
    return predictor


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚽ FootPredict-Pro")
st.sidebar.markdown("*State-of-the-art football prediction powered by Dixon-Coles + XGBoost ensemble*")
st.sidebar.divider()

# Team inputs
st.sidebar.subheader("Match Setup")
home_team = st.sidebar.text_input("🏠 Home Team", value="Manchester City")
away_team = st.sidebar.text_input("✈️ Away Team", value="Arsenal")

st.sidebar.divider()

# Lineup inputs
st.sidebar.subheader("Starting Lineups (optional)")
home_lineup_str = st.sidebar.text_area(
    "Home XI (comma-separated)",
    value="Ederson,Walker,Dias,Akanji,Gvardiol,Rodri,De Bruyne,Silva,Doku,Foden,Haaland",
    height=80,
)
away_lineup_str = st.sidebar.text_area(
    "Away XI (comma-separated)",
    value="Raya,White,Saliba,Gabriel,Zinchenko,Partey,Rice,Odegaard,Saka,Havertz,Martinelli",
    height=80,
)

predict_button = st.sidebar.button("🔮 Predict Match", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("⚽ FootPredict-Pro")
st.markdown("### AI-Powered Football Match Predictor")
st.markdown(
    "Combining **Dixon-Coles bivariate Poisson** + **XGBoost/LightGBM/CatBoost ensemble** "
    "for calibrated match outcome and goal-scorer predictions."
)
st.divider()

if predict_button or True:  # Show placeholder on load
    with st.spinner("Generating prediction..."):
        try:
            predictor = load_predictor()

            home_lineup = [p.strip() for p in home_lineup_str.split(",") if p.strip()] or None
            away_lineup = [p.strip() for p in away_lineup_str.split(",") if p.strip()] or None

            pred = predictor.predict(
                home_team=home_team,
                away_team=away_team,
                home_lineup=home_lineup,
                away_lineup=away_lineup,
            )

            # --- Match header ---
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"## 🏠 {pred.home_team}")
                st.metric("xG", f"{pred.home_xg:.2f}")
            with col2:
                st.markdown("### VS")
            with col3:
                st.markdown(f"## ✈️ {pred.away_team}")
                st.metric("xG", f"{pred.away_xg:.2f}")

            st.divider()

            # --- Outcome probabilities ---
            st.subheader("📊 Match Outcome Probabilities")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"{pred.home_team} Win",
                    f"{pred.p_home_win:.1%}",
                    delta=f"Poisson: {pred.poisson_p_home:.1%}",
                )
            with col2:
                st.metric(
                    "Draw",
                    f"{pred.p_draw:.1%}",
                    delta=f"Poisson: {pred.poisson_p_draw:.1%}",
                )
            with col3:
                st.metric(
                    f"{pred.away_team} Win",
                    f"{pred.p_away_win:.1%}",
                    delta=f"Poisson: {pred.poisson_p_away:.1%}",
                )

            # Probability bar chart
            import pandas as pd
            prob_df = pd.DataFrame({
                "Outcome": [f"{pred.home_team} Win", "Draw", f"{pred.away_team} Win"],
                "Probability": [pred.p_home_win, pred.p_draw, pred.p_away_win],
            })
            st.bar_chart(prob_df.set_index("Outcome"), use_container_width=True)

            st.divider()

            # --- Top scorelines ---
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Top Scorelines")
                score_data = []
                for h, a, p in pred.top_scorelines:
                    score_data.append({
                        "Score": f"{h} - {a}",
                        "Probability": f"{p:.1%}",
                    })
                st.table(pd.DataFrame(score_data))

            with col2:
                # --- Player predictions ---
                st.subheader("⚽ Goal Scorer Predictions")

                if pred.home_top_scorer:
                    s = pred.home_top_scorer
                    st.metric(
                        f"🏠 {s.name} ({pred.home_team})",
                        f"P(≥1 goal): {s.p_goal:.1%}",
                        delta=f"xG λ = {s.lambda_xg:.3f}",
                    )

                if pred.away_top_scorer:
                    s = pred.away_top_scorer
                    st.metric(
                        f"✈️ {s.name} ({pred.away_team})",
                        f"P(≥1 goal): {s.p_goal:.1%}",
                        delta=f"xG λ = {s.lambda_xg:.3f}",
                    )

                if not pred.home_top_scorer and not pred.away_top_scorer:
                    st.info("Add lineups in the sidebar to see player goal probabilities.")

            st.divider()

            # --- Full lineup table ---
            if pred.all_home_players or pred.all_away_players:
                st.subheader("📋 Full Lineup Predictions")
                lcol, rcol = st.columns(2)

                if pred.all_home_players:
                    with lcol:
                        st.markdown(f"**{pred.home_team}**")
                        home_data = [
                            {
                                "Player": p.name,
                                "Position": p.position or "—",
                                "P(Goal)": f"{p.p_goal:.1%}",
                                "xG λ": f"{p.lambda_xg:.3f}",
                            }
                            for p in pred.all_home_players
                        ]
                        st.dataframe(pd.DataFrame(home_data), hide_index=True, use_container_width=True)

                if pred.all_away_players:
                    with rcol:
                        st.markdown(f"**{pred.away_team}**")
                        away_data = [
                            {
                                "Player": p.name,
                                "Position": p.position or "—",
                                "P(Goal)": f"{p.p_goal:.1%}",
                                "xG λ": f"{p.lambda_xg:.3f}",
                            }
                            for p in pred.all_away_players
                        ]
                        st.dataframe(pd.DataFrame(away_data), hide_index=True, use_container_width=True)

            # --- Metadata ---
            st.divider()
            confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                pred.confidence, "⚪"
            )
            st.markdown(
                f"*{confidence_color} Confidence: **{pred.confidence}** | "
                f"⏱️ Inference: {pred.inference_time_ms:.0f}ms*"
            )

            # JSON export
            with st.expander("📄 Export as JSON"):
                st.code(json.dumps(pred.to_dict(), indent=2), language="json")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "*FootPredict-Pro — Dixon-Coles + XGBoost Ensemble | "
    "RPS < 0.20 | Brier < 0.22*",
    unsafe_allow_html=False,
)
