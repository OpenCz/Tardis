import datetime
import glob
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="TARDIS · Retards TGV",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET = "Average delay of all trains at arrival"
SEASON_MAP = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}
SEASON_EMOJIS = {"winter": "❄️", "spring": "🌸", "summer": "☀️", "autumn": "🍂"}
ROUTE_STAT_FEATURES = [
    "Average journey time",
    "Number of scheduled trains",
    "Number of cancelled trains",
    "cancellation_rate",
]
NUMERIC_FEATURES = ROUTE_STAT_FEATURES + ["year", "month", "day_of_week"]
CATEGORICAL_FEATURES = ["Departure station", "Arrival station", "Service", "season"]

C = {
    "primary": "#4f46e5",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "orange": "#f97316",
    "neutral": "#64748b",
    "bg": "#f8fafc",
    "card": "#ffffff",
    "border": "#e2e8f0",
    "sidebar": "#0f172a",
    "text": "#0f172a",
    "muted": "#64748b",
}


def delay_info(v: float) -> tuple:
    if v < 5:
        return (
            "A l'heure",
            "#16a34a",
            "#f0fdf4",
            "Ton train devrait arriver sans souci.",
        )
    if v < 15:
        return (
            "Petit retard",
            "#d97706",
            "#fffbeb",
            "Un leger retard, reste a portee du quai.",
        )
    if v < 30:
        return (
            "Retard modere",
            "#f97316",
            "#fff7ed",
            "Prevois de quoi patienter sur le quai.",
        )
    return (
        "Retard important",
        "#dc2626",
        "#fef2f2",
        "Mieux vaut avoir de la batterie et de quoi lire.",
    )


def delay_info_en(v: float) -> tuple:
    if v < 5:
        return (
            "On time",
            "#16a34a",
            "#f0fdf4",
            "Your train should arrive without a hitch.",
        )
    if v < 15:
        return (
            "Slight delay",
            "#d97706",
            "#fffbeb",
            "Just a small delay, stay near the platform.",
        )
    if v < 30:
        return (
            "Moderate delay",
            "#f97316",
            "#fff7ed",
            "Bring something to pass the time.",
        )
    return (
        "Significant delay",
        "#dc2626",
        "#fef2f2",
        "Better have battery and something to read.",
    )


TR = {
    "fr": {
        "lang_btn": "English",
        "nav_title": "Menu",
        "nav_predict": "Predire un retard",
        "nav_explore": "Voir les stats",
        "nav_models": "Comment ca marche ?",
        "model_title": "Modele IA",
        "model_help": "Algorithme utilise pour la prediction",
        "model_one_only": "Un seul modele disponible. Relancez le notebook pour en generer plusieurs.",
        "filters_title": "Filtres",
        "f_stations": "Gares de depart",
        "f_years": "Periode",
        "f_seasons": "Saisons",
        "dataset_info": "trajets dans la base",
        "s_winter": "Hiver",
        "s_spring": "Printemps",
        "s_summer": "Ete",
        "s_autumn": "Automne",
        "p_title": "Predire le retard de ton train",
        "p_sub": "Choisis ta gare de depart, d'arrivee et la date — l'IA fait le reste.",
        "p_dep": "Gare de depart",
        "p_arr": "Gare d'arrivee",
        "p_date": "Date du voyage",
        "p_btn": "Calculer mon retard estime",
        "p_approx": "Trajet non reference — estimation via des trajets similaires",
        "p_min": "minutes de retard estime",
        "p_hist": "Moyenne habituelle sur ce trajet",
        "p_sigma": "Impredictibilite",
        "p_sigma_help": "Plus ce chiffre est grand, plus les retards sont impredictibles sur ce trajet.",
        "p_by": "Prediction par **{m}**",
        "p_dist_title": "Historique des retards sur ce trajet",
        "p_pred_marker": "Prediction : {v:.1f} min",
        "p_avg_marker": "Moyenne habituelle",
        "p_compare_title": "Retard estime selon la saison",
        "p_placeholder": "Choisis tes gares et une date, puis clique sur le bouton.",
        "e_title": "Les retards en chiffres",
        "e_sub": "Toutes les statistiques sur les retards TGV SNCF.",
        "e_k_records": "Trajets analyses",
        "e_k_delay": "Retard moyen",
        "e_k_punct": "A l'heure",
        "e_k_cancel": "Trains annules",
        "e_dist": "Repartition des retards",
        "e_trend": "Evolution dans le temps",
        "e_stations": "Top 15 des gares les plus en retard",
        "e_heatmap": "Retard moyen par saison et annee",
        "e_export": "Telecharger les donnees",
        "e_dl_btn": "Exporter en CSV",
        "e_avg": "Moyenne",
        "e_delay_ax": "Retard (min)",
        "e_count_ax": "Nombre de trajets",
        "m_title": "Comment ca marche ?",
        "m_sub": "L'IA analyse des millions de trajets reels pour predire ton retard.",
        "m_step1_title": "1. Tu choisis ton trajet",
        "m_step1_body": "Deux gares et une date. C'est tout ce dont tu as besoin de fournir.",
        "m_step2_title": "2. L'IA creuse dans l'historique",
        "m_step2_body": "Elle analyse ce qui s'est passe sur ce trajet par le passe : duree moyenne, taux d'annulation, influence des saisons, jour de la semaine...",
        "m_step3_title": "3. Elle te donne une estimation",
        "m_step3_body": "La prediction est basee sur des centaines de milliers de trajets reels enregistres par la SNCF depuis 2018.",
        "m_accuracy_exp": "En moyenne, la prediction se trompe de **{rmse:.1f} minutes**.",
        "m_imp_title": "Qu'est-ce qui influence le plus le retard ?",
        "m_imp_na": "Information non disponible pour ce modele.",
        "m_table_title": "Comparaison des modeles IA disponibles",
        "m_rmse": "Erreur moy. (min)",
        "m_r2": "Precision (R2)",
    },
    "en": {
        "lang_btn": "Francais",
        "nav_title": "Menu",
        "nav_predict": "Predict a delay",
        "nav_explore": "See stats",
        "nav_models": "How it works",
        "model_title": "AI model",
        "model_help": "Algorithm used for prediction",
        "model_one_only": "Only one model available. Re-run the notebook to generate more.",
        "filters_title": "Filters",
        "f_stations": "Departure stations",
        "f_years": "Year range",
        "f_seasons": "Seasons",
        "dataset_info": "journeys in the database",
        "s_winter": "Winter",
        "s_spring": "Spring",
        "s_summer": "Summer",
        "s_autumn": "Autumn",
        "p_title": "Predict your train delay",
        "p_sub": "Pick your departure, arrival and date — AI handles the rest.",
        "p_dep": "Departure station",
        "p_arr": "Arrival station",
        "p_date": "Travel date",
        "p_btn": "Calculate my estimated delay",
        "p_approx": "Unknown route — estimate based on similar journeys",
        "p_min": "minutes estimated delay",
        "p_hist": "Usual average on this route",
        "p_sigma": "Unpredictability",
        "p_sigma_help": "Higher number = more unpredictable delays on this route.",
        "p_by": "Predicted by **{m}**",
        "p_dist_title": "Historical delays on this route",
        "p_pred_marker": "Prediction: {v:.1f} min",
        "p_avg_marker": "Usual average",
        "p_compare_title": "Estimated delay by season",
        "p_placeholder": "Pick your stations and a date, then click the button.",
        "e_title": "Delays in numbers",
        "e_sub": "All statistics about TGV SNCF delays.",
        "e_k_records": "Journeys analysed",
        "e_k_delay": "Average delay",
        "e_k_punct": "On time",
        "e_k_cancel": "Cancelled trains",
        "e_dist": "Delay distribution",
        "e_trend": "Trend over time",
        "e_stations": "Top 15 most delayed departure stations",
        "e_heatmap": "Average delay by season and year",
        "e_export": "Download data",
        "e_dl_btn": "Export CSV",
        "e_avg": "Average",
        "e_delay_ax": "Delay (min)",
        "e_count_ax": "Number of journeys",
        "m_title": "How does it work?",
        "m_sub": "The AI analyses millions of real journeys to predict your delay.",
        "m_step1_title": "1. You choose your journey",
        "m_step1_body": "Two stations and a date. That's all you need to provide.",
        "m_step2_title": "2. AI digs through history",
        "m_step2_body": "It analyses what happened on this route in the past: average duration, cancellation rate, seasonal influence, day of week...",
        "m_step3_title": "3. It gives you an estimate",
        "m_step3_body": "The prediction is based on hundreds of thousands of real journeys recorded by SNCF since 2018.",
        "m_accuracy_exp": "On average, the prediction is off by **{rmse:.1f} minutes**.",
        "m_imp_title": "What influences delays the most?",
        "m_imp_na": "Information not available for this model type.",
        "m_table_title": "Comparison of available AI models",
        "m_rmse": "Avg error (min)",
        "m_r2": "Accuracy (R2)",
    },
}

for k, v in [("lang", "fr"), ("page", "predict"), ("prediction", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


def t(key: str, **kw) -> str:
    txt = TR[st.session_state.lang].get(key, key)
    return txt.format(**kw) if kw else txt


def slabel(sk: str) -> str:
    return t(f"s_{sk}")


def get_delay_info(v: float):
    return (delay_info_en if st.session_state.lang == "en" else delay_info)(v)


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.main .block-container {{ padding: 2rem 2.5rem 3rem; max-width: 1200px; }}

[data-testid="stSidebar"] {{ background: {C["sidebar"]} !important; }}
[data-testid="stSidebar"] > div {{ padding: 1.5rem 1rem; }}
[data-testid="stSidebar"] * {{ color: #cbd5e1 !important; }}
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stSlider > label {{
    color: #94a3b8 !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; border: none !important;
    color: #94a3b8 !important; text-align: left !important;
    padding: 0.6rem 0.75rem !important; border-radius: 8px !important;
    width: 100% !important; font-size: 0.9rem !important;
    font-weight: 500 !important; transition: all 0.15s !important;
    margin-bottom: 3px !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: #1e293b !important; color: #f1f5f9 !important;
}}
[data-testid="stSidebar"] hr {{ border-color: #1e293b !important; margin: 1rem 0 !important; }}

.ph {{ margin-bottom: 2rem; }}
.ph-title {{ font-size: 1.75rem; font-weight: 800; color: {C["text"]}; margin: 0; letter-spacing: -0.02em; }}
.ph-sub   {{ font-size: 0.9rem; color: {C["muted"]}; margin: 0.4rem 0 0; }}

.kpi {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 14px; padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.05);
}}
.kpi-val {{ font-size: 2.2rem; font-weight: 800; color: {C["text"]}; line-height: 1.1; margin: 0.25rem 0 0.15rem; }}
.kpi-lbl {{ font-size: 0.7rem; font-weight: 700; color: {C["muted"]}; text-transform: uppercase; letter-spacing: 0.08em; }}
.kpi-sub {{ font-size: 0.78rem; color: #94a3b8; }}

.result-card {{
    border-radius: 20px; padding: 2.5rem 2rem; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.12);
}}
.result-val   {{ font-size: 6rem; font-weight: 900; line-height: 1; letter-spacing: -0.04em; }}
.result-unit  {{ font-size: 1.1rem; font-weight: 500; opacity: 0.75; margin-top: 0.2rem; }}
.result-label {{ font-size: 1.4rem; font-weight: 700; margin-top: 0.75rem; }}
.result-msg   {{ font-size: 1rem; font-weight: 400; opacity: 0.85; margin-top: 0.4rem; }}
.result-route {{ font-size: 0.85rem; font-weight: 500; opacity: 0.6; margin-bottom: 0.75rem; }}

.step-card {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 14px; padding: 1.5rem; height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}}
.step-title {{ font-size: 1.05rem; font-weight: 700; color: {C["text"]}; margin-bottom: 0.5rem; }}
.step-body  {{ font-size: 0.88rem; color: {C["muted"]}; line-height: 1.6; margin: 0; }}

.stitle {{
    font-size: 0.8rem; font-weight: 700; color: {C["muted"]};
    text-transform: uppercase; letter-spacing: 0.07em; margin: 0 0 0.75rem;
}}

.placeholder {{
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    min-height: 340px; background: #f8fafc; border: 2px dashed {C["border"]};
    border-radius: 20px; text-align: center; padding: 2rem;
}}
.placeholder p {{ color: {C["muted"]}; font-size: 1rem; margin: 0; font-weight: 500; }}

.acc-banner {{
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px; padding: 1.5rem 2rem; color: white; text-align: center;
}}
.acc-banner-val {{ font-size: 3rem; font-weight: 900; letter-spacing: -0.03em; }}
.acc-banner-lbl {{ font-size: 0.9rem; opacity: 0.8; margin-top: 0.25rem; }}

.sep {{ height: 1px; background: {C["border"]}; margin: 2rem 0; border: none; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/processed/trains/cleaned_dataset.csv", parse_dates=["Date"])
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["route"] = df["Departure station"] + " → " + df["Arrival station"]
    return df


@st.cache_data
def build_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Departure station", "Arrival station"])[
            ROUTE_STAT_FEATURES + ["Service"]
        ]
        .agg({f: "median" for f in ROUTE_STAT_FEATURES} | {"Service": "first"})
        .reset_index()
    )


def discover_models() -> dict:
    meta: dict = {}
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json") as f:
            meta = json.load(f)
    catalog: dict = {}
    for path in sorted(glob.glob("models/*.joblib")):
        try:
            art = joblib.load(path)
            name = (
                art.get("model_name", os.path.basename(path))
                if isinstance(art, dict)
                else os.path.basename(path)
            )
            catalog[name] = {"file": path, **meta.get(name, {})}
        except Exception:
            pass
    if os.path.exists("model.joblib"):
        try:
            art = joblib.load("model.joblib")
            name = (
                art.get("model_name", "Modele") if isinstance(art, dict) else "Modele"
            )
            if name not in catalog:
                catalog[name] = {"file": "model.joblib", **meta.get(name, {})}
        except Exception:
            pass
    return catalog


@st.cache_resource
def load_pipeline(model_file: str):
    art = joblib.load(model_file)
    if isinstance(art, dict):
        return art["pipeline"], art["model_name"]
    return art, type(art.named_steps["model"]).__name__


@st.cache_data
def get_importance(_pipeline) -> pd.Series:
    try:
        est = _pipeline.named_steps["model"]
        cat_enc = (
            _pipeline.named_steps["prep"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_names = cat_enc.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        names = NUMERIC_FEATURES + cat_names
        imp = getattr(
            est, "feature_importances_", getattr(est, "coef_", np.zeros(len(names)))
        )
        return pd.Series(np.abs(imp), index=names).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)


def predict(dep: str, arr: str, date, pipeline, route_stats: pd.DataFrame):
    row = route_stats[
        (route_stats["Departure station"] == dep)
        & (route_stats["Arrival station"] == arr)
    ]
    approx = False
    if row.empty:
        row = route_stats[route_stats["Departure station"] == dep]
        approx = True
    if row.empty:
        row = route_stats
        approx = True
    if row.empty:
        return None, False
    stats = row[ROUTE_STAT_FEATURES].mean()
    service = row["Service"].mode().iloc[0]
    m = date.month
    inp = pd.DataFrame(
        [
            {
                "Average journey time": stats["Average journey time"],
                "Number of scheduled trains": stats["Number of scheduled trains"],
                "Number of cancelled trains": stats["Number of cancelled trains"],
                "cancellation_rate": stats["cancellation_rate"],
                "year": date.year,
                "month": m,
                "day_of_week": date.weekday(),
                "Departure station": dep,
                "Arrival station": arr,
                "Service": service,
                "season": SEASON_MAP[m],
            }
        ]
    )
    return max(0.0, float(pipeline.predict(inp)[0])), approx


def chart_style(fig, height: int = 320):
    fig.update_layout(
        height=height,
        margin=dict(t=20, b=20, l=10, r=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", size=11, color=C["text"]),
        xaxis=dict(gridcolor="#f1f5f9", linecolor=C["border"]),
        yaxis=dict(gridcolor="#f1f5f9", linecolor=C["border"]),
    )
    return fig


def make_gauge(v: float, max_val: float = 60) -> go.Figure:
    _, color, _, _ = get_delay_info(v)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(v, 1),
            number={
                "suffix": " min",
                "font": {"size": 36, "family": "Inter", "color": C["text"]},
            },
            gauge={
                "axis": {
                    "range": [0, max_val],
                    "tickwidth": 1,
                    "tickcolor": C["muted"],
                    "tickfont": {"size": 10},
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 5], "color": "#dcfce7"},
                    {"range": [5, 15], "color": "#fef9c3"},
                    {"range": [15, 30], "color": "#ffedd5"},
                    {"range": [30, max_val], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": v,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=20, r=20),
        paper_bgcolor="white",
        font=dict(family="Inter"),
    )
    return fig


df = load_data()
route_stats = build_route_stats(df)
stations = sorted(df["Departure station"].dropna().unique())
catalog = discover_models()

with st.sidebar:
    st.markdown("## TARDIS")
    st.caption("Predicteur de retards TGV SNCF")

    if st.button(t("lang_btn"), width="stretch"):
        st.session_state.lang = "en" if st.session_state.lang == "fr" else "fr"
        st.rerun()

    st.divider()
    st.markdown(f"**{t('nav_title')}**")

    for pid, lbl in [
        ("predict", t("nav_predict")),
        ("explore", t("nav_explore")),
        ("models", t("nav_models")),
    ]:
        if st.button(lbl, key=f"nav_{pid}", width="stretch"):
            st.session_state.page = pid
            st.rerun()

    st.divider()
    st.markdown(f"**{t('model_title')}**")
    if not catalog:
        st.warning(t("model_one_only"))
        st.stop()

    model_names = list(catalog.keys())
    if len(model_names) == 1:
        st.info(t("model_one_only"))

    def fmt_model(n):
        m = catalog[n]
        return f"{n}  (±{m['RMSE']:.1f} min)" if "RMSE" in m else n

    sel_model = st.selectbox(
        "_",
        options=model_names,
        format_func=fmt_model,
        label_visibility="collapsed",
        help=t("model_help"),
        key="model_sel",
    )
    pipeline, model_name = load_pipeline(catalog[sel_model]["file"])
    model_meta = catalog[sel_model]
    importance = get_importance(pipeline)

    st.divider()
    st.markdown(f"**{t('filters_title')}**")
    sel_stations = st.multiselect(t("f_stations"), stations, default=list(stations[:8]))
    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider(t("f_years"), ymin, ymax, (ymin, ymax))

    SEASON_OPTS = {
        t("s_winter"): "winter",
        t("s_spring"): "spring",
        t("s_summer"): "summer",
        t("s_autumn"): "autumn",
    }
    sel_s_labels = st.multiselect(
        t("f_seasons"), list(SEASON_OPTS), default=list(SEASON_OPTS)
    )
    sel_seasons = [SEASON_OPTS[s] for s in sel_s_labels] or list(SEASON_OPTS.values())

    st.divider()
    st.caption(f"{len(df):,} {t('dataset_info')}")

df_f = df[
    df["Departure station"].isin(sel_stations or stations)
    & df["year"].between(*year_range)
    & df["season"].isin(sel_seasons)
]

page = st.session_state.page


if page == "predict":
    st.markdown(
        f'<div class="ph"><p class="ph-title">{t("p_title")}</p>'
        f'<p class="ph-sub">{t("p_sub")}</p></div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        dep = st.selectbox(t("p_dep"), stations, key="dep_sel")
        arr_opts = sorted(
            route_stats[route_stats["Departure station"] == dep]["Arrival station"]
        ) or list(stations)
        arr = st.selectbox(t("p_arr"), arr_opts, key="arr_sel")
        date = st.date_input(
            t("p_date"),
            value=datetime.date.today(),
            min_value=datetime.date(2018, 1, 1),
        )

        if st.button(t("p_btn"), type="primary", width="stretch"):
            result, approx = predict(dep, arr, date, pipeline, route_stats)
            st.session_state.prediction = {
                "dep": dep,
                "arr": arr,
                "date": date,
                "result": result,
                "approx": approx,
                "model": model_name,
            }

        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.78rem;margin-top:0.5rem">{t("p_by", m=model_name)}</p>',
            unsafe_allow_html=True,
        )

    with result_col:
        pred_data = st.session_state.get("prediction")

        if not pred_data:
            st.markdown(
                f'<div class="placeholder"><p>{t("p_placeholder")}</p></div>',
                unsafe_allow_html=True,
            )
        elif pred_data["result"] is None:
            st.error(t("p_approx"))
        else:
            v = pred_data["result"]
            dep_ = pred_data["dep"]
            arr_ = pred_data["arr"]
            date_ = pred_data["date"]
            approx_ = pred_data.get("approx", False)

            label, color, bg, msg = get_delay_info(v)

            st.markdown(
                f"""
            <div class="result-card" style="background:{bg}; color:{color};">
                <div class="result-route" style="color:{color}">{dep_} &rarr; {arr_}</div>
                <div class="result-val"   style="color:{color}">{v:.0f}</div>
                <div class="result-unit"  style="color:{color}">{t("p_min")}</div>
                <div class="result-label">{label}</div>
                <div class="result-msg"   style="color:{color}">{msg}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if approx_:
                st.caption(t("p_approx"))

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(make_gauge(v), width="stretch")

            hist = df[
                (df["Departure station"] == dep_) & (df["Arrival station"] == arr_)
            ][TARGET].dropna()
            h_mean = hist.mean() if len(hist) else None
            h_std = hist.std() if len(hist) else None

            if h_mean is not None:
                mc1, mc2 = st.columns(2)
                mc1.metric(
                    t("p_hist"),
                    f"{h_mean:.1f} min",
                    delta=f"{v - h_mean:+.1f} min",
                    delta_color="inverse",
                )
                if h_std is not None:
                    mc2.metric(
                        t("p_sigma"), f"±{h_std:.1f} min", help=t("p_sigma_help")
                    )

    pred_data = st.session_state.get("prediction")
    if pred_data and pred_data["result"] is not None:
        v, dep_, arr_, date_ = (
            pred_data["result"],
            pred_data["dep"],
            pred_data["arr"],
            pred_data["date"],
        )
        hist = df[(df["Departure station"] == dep_) & (df["Arrival station"] == arr_)][
            TARGET
        ].dropna()
        h_mean = hist.mean() if len(hist) else None

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        left2, right2 = st.columns(2, gap="large")

        with left2:
            if len(hist) >= 5:
                st.markdown(
                    f'<p class="stitle">{t("p_dist_title")}</p>', unsafe_allow_html=True
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=hist,
                        nbinsx=35,
                        marker_color=C["primary"],
                        opacity=0.75,
                        name="",
                    )
                )
                _, color, _, _ = get_delay_info(v)
                fig.add_vline(
                    x=v,
                    line_color=color,
                    line_width=2.5,
                    annotation_text=t("p_pred_marker", v=v),
                    annotation_font_color=color,
                )
                if h_mean:
                    fig.add_vline(
                        x=h_mean,
                        line_color=C["success"],
                        line_dash="dot",
                        line_width=1.5,
                        annotation_text=t("p_avg_marker"),
                        annotation_font_color=C["success"],
                    )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title=t("e_delay_ax"),
                    yaxis_title=t("e_count_ax"),
                )
                st.plotly_chart(chart_style(fig, 300), width="stretch")

        with right2:
            st.markdown(
                f'<p class="stitle">{t("p_compare_title")}</p>', unsafe_allow_html=True
            )
            rows = []
            for sk2, m2 in {
                "winter": 1,
                "spring": 4,
                "summer": 7,
                "autumn": 10,
            }.items():
                try:
                    d2 = date_.replace(month=m2)
                except ValueError:
                    d2 = date_.replace(month=m2, day=28)
                p2, _ = predict(dep_, arr_, d2, pipeline, route_stats)
                if p2 is not None:
                    _, col2, _, _ = get_delay_info(p2)
                    rows.append(
                        {
                            "Saison": f"{SEASON_EMOJIS[sk2]} {slabel(sk2)}",
                            t("e_delay_ax"): round(p2, 1),
                            "color": col2,
                        }
                    )
            if rows:
                cdf = pd.DataFrame(rows)
                fig2 = px.bar(
                    cdf,
                    x="Saison",
                    y=t("e_delay_ax"),
                    color="color",
                    color_discrete_map="identity",
                    text=t("e_delay_ax"),
                )
                fig2.update_traces(textposition="outside", textfont_size=13)
                fig2.update_layout(
                    coloraxis_showscale=False,
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title=t("e_delay_ax"),
                )
                st.plotly_chart(chart_style(fig2, 300), width="stretch")


elif page == "explore":
    st.markdown(
        f'<div class="ph"><p class="ph-title">{t("e_title")}</p>'
        f'<p class="ph-sub">{t("e_sub")}</p></div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    for col, lbl, val, sub in [
        (k1, t("e_k_records"), f"{len(df_f):,}", f"/ {len(df):,} total"),
        (k2, t("e_k_delay"), f"{df_f[TARGET].mean():.1f} min", "retard moyen arrivee"),
        (
            k3,
            t("e_k_punct"),
            f"{df_f['punctuality_rate'].mean() * 100:.1f} %",
            "trains a l'heure",
        ),
        (
            k4,
            t("e_k_cancel"),
            f"{df_f['cancellation_rate'].mean() * 100:.2f} %",
            "trains annules",
        ),
    ]:
        col.markdown(
            f'<div class="kpi"><p class="kpi-lbl">{lbl}</p>'
            f'<p class="kpi-val">{val}</p><p class="kpi-sub">{sub}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f'<p class="stitle">{t("e_dist")}</p>', unsafe_allow_html=True)
        fig = px.histogram(
            df_f.dropna(subset=[TARGET]),
            x=TARGET,
            nbins=60,
            color_discrete_sequence=[C["primary"]],
            labels={TARGET: t("e_delay_ax")},
        )
        fig.update_layout(
            showlegend=False,
            bargap=0.05,
            xaxis_title=t("e_delay_ax"),
            yaxis_title=t("e_count_ax"),
        )
        st.plotly_chart(chart_style(fig), width="stretch")

    with c2:
        st.markdown(f'<p class="stitle">{t("e_trend")}</p>', unsafe_allow_html=True)
        mon = df_f.groupby(["year", "month"])[TARGET].mean().reset_index()
        mon["period"] = pd.to_datetime(mon[["year", "month"]].assign(day=1))
        fig2 = px.line(
            mon.sort_values("period"),
            x="period",
            y=TARGET,
            color_discrete_sequence=[C["primary"]],
            labels={TARGET: t("e_delay_ax"), "period": ""},
        )
        mv = df_f[TARGET].mean()
        fig2.add_hline(
            y=mv,
            line_dash="dot",
            line_color="#94a3b8",
            annotation_text=f"{t('e_avg')} {mv:.1f} min",
            annotation_font_color="#94a3b8",
        )
        fig2.update_layout(xaxis_title="", yaxis_title=t("e_delay_ax"))
        st.plotly_chart(chart_style(fig2), width="stretch")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown(f'<p class="stitle">{t("e_stations")}</p>', unsafe_allow_html=True)
        top = (
            df_f.groupby("Departure station")[TARGET]
            .mean()
            .dropna()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig3 = px.bar(
            top,
            x=TARGET,
            y="Departure station",
            orientation="h",
            color=TARGET,
            color_continuous_scale="RdYlGn_r",
            labels={TARGET: t("e_delay_ax"), "Departure station": ""},
        )
        fig3.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis_title=t("e_delay_ax"),
        )
        st.plotly_chart(chart_style(fig3, 420), width="stretch")

    with c4:
        st.markdown(f'<p class="stitle">{t("e_heatmap")}</p>', unsafe_allow_html=True)
        pivot = df_f.pivot_table(
            values=TARGET, index="year", columns="season", aggfunc="mean"
        )
        pivot = pivot.reindex(columns=["winter", "spring", "summer", "autumn"])
        pivot.columns = [slabel(c) for c in pivot.columns]
        fig4 = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="RdYlGn_r",
            labels={"color": t("e_delay_ax")},
        )
        st.plotly_chart(chart_style(fig4, 420), width="stretch")

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown(f'<p class="stitle">{t("e_export")}</p>', unsafe_allow_html=True)
    buf = io.BytesIO()
    df_f.to_csv(buf, index=False)
    st.download_button(
        t("e_dl_btn"),
        data=buf.getvalue(),
        file_name="tardis_export.csv",
        mime="text/csv",
        width="stretch",
    )


elif page == "models":
    st.markdown(
        f'<div class="ph"><p class="ph-title">{t("m_title")}</p>'
        f'<p class="ph-sub">{t("m_sub")}</p></div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3, gap="medium")
    for col, tk, bk in [
        (s1, "m_step1_title", "m_step1_body"),
        (s2, "m_step2_title", "m_step2_body"),
        (s3, "m_step3_title", "m_step3_body"),
    ]:
        col.markdown(
            f'<div class="step-card"><p class="step-title">{t(tk)}</p>'
            f'<p class="step-body">{t(bk)}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if "RMSE" in model_meta:
        rmse = model_meta["RMSE"]
        r2 = model_meta.get("R2", 0)
        acc_pct = max(0, min(100, r2 * 100))

        ac1, ac2, ac3 = st.columns(3)
        ac1.markdown(
            f'<div class="acc-banner">'
            f'<div class="acc-banner-val">±{rmse:.1f}</div>'
            f'<div class="acc-banner-lbl">{t("e_delay_ax")} — erreur moyenne</div></div>',
            unsafe_allow_html=True,
        )
        ac2.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,#059669,#10b981)">'
            f'<div class="acc-banner-val">{acc_pct:.0f}%</div>'
            f'<div class="acc-banner-lbl">de variance expliquee (R2)</div></div>',
            unsafe_allow_html=True,
        )
        ac3.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,#0ea5e9,#6366f1)">'
            f'<div class="acc-banner-val">{model_name.split()[0]}</div>'
            f'<div class="acc-banner-lbl">modele actif</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(t("m_accuracy_exp", rmse=rmse))

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        st.markdown(f'<p class="stitle">{t("m_imp_title")}</p>', unsafe_allow_html=True)
        if not importance.empty:
            FEATURE_LABELS = {
                "Average journey time": "Duree du trajet",
                "Number of scheduled trains": "Trains programmes",
                "Number of cancelled trains": "Trains annules",
                "cancellation_rate": "Taux d'annulation",
                "month": "Mois de l'annee",
                "day_of_week": "Jour de la semaine",
                "year": "Annee",
            }
            top15 = importance.head(15).sort_values()
            labels = [
                FEATURE_LABELS.get(
                    i,
                    i.replace("_", " ")
                    .replace("Departure station_", "Depart : ")
                    .replace("Arrival station_", "Arrivee : "),
                )
                for i in top15.index
            ]
            colors = [
                C["primary"] if importance[i] > importance.median() else "#a5b4fc"
                for i in top15.index
            ]
            fig_imp = go.Figure(
                go.Bar(
                    x=top15.values,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in top15.values],
                    textposition="outside",
                )
            )
            fig_imp.update_layout(
                xaxis_title="Importance relative", yaxis_title="", showlegend=False
            )
            st.plotly_chart(chart_style(fig_imp, 500), width="stretch")
        else:
            st.info(t("m_imp_na"))

    with right:
        if len(catalog) > 1:
            st.markdown(
                f'<p class="stitle">{t("m_table_title")}</p>', unsafe_allow_html=True
            )
            rows_m = [
                {"Modele": name, t("m_rmse"): meta["RMSE"], t("m_r2"): meta["R2"]}
                for name, meta in catalog.items()
                if all(k in meta for k in ("RMSE", "MAE", "R2"))
            ]
            if rows_m:
                perf = pd.DataFrame(rows_m).sort_values(t("m_rmse"))
                best_idx = perf[t("m_rmse")].idxmin()

                def highlight(row):
                    return (
                        ["background-color:#f0fdf4;font-weight:700"] * len(row)
                        if row.name == best_idx
                        else [""] * len(row)
                    )

                st.dataframe(
                    perf.style.apply(highlight, axis=1)
                    .format({t("m_rmse"): "±{:.1f} min", t("m_r2"): "{:.1%}"})
                    .bar(subset=[t("m_rmse")], color="#fecaca", vmin=0)
                    .bar(subset=[t("m_r2")], color="#bbf7d0", vmin=0, vmax=1),
                    width="stretch",
                    hide_index=True,
                )
                best_row = perf.iloc[0]
                st.success(
                    f"Meilleur modele : {best_row['Modele']} — erreur moyenne ±{best_row[t('m_rmse')]:.1f} min"
                )
