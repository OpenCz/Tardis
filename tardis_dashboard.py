"""
TARDIS · Tableau de bord des retards TGV SNCF — v2
Redesigned: visual identity, ergonomic filters, rich charts.
"""

import datetime
import glob
import io
import json
import os

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TARDIS · Retards TGV",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET      = "Average delay of all trains at arrival"
DEP_TARGET  = "Average delay of all trains at departure"

SEASON_MAP = {
    1: "winter", 2: "winter",  3: "spring", 4: "spring",  5: "spring",
    6: "summer", 7: "summer",  8: "summer", 9: "autumn", 10: "autumn",
    11: "autumn", 12: "winter",
}
SEASON_EMOJIS = {"winter": "❄️", "spring": "🌸", "summer": "☀️", "autumn": "🍂"}

CAUSE_COLS = [
    ("Pct delay due to external causes",
     "Causes externes", "External causes", "#ef4444"),
    ("Pct delay due to infrastructure",
     "Infrastructure", "Infrastructure", "#f97316"),
    ("Pct delay due to traffic management",
     "Gestion trafic", "Traffic mgmt", "#f59e0b"),
    ("Pct delay due to rolling stock",
     "Matériel roulant", "Rolling stock", "#eab308"),
    ("Pct delay due to station management and equipment reuse",
     "Gestion gare", "Station management", "#84cc16"),
    ("Pct delay due to passenger handling (crowding, disabled persons, connections)",
     "Passagers", "Passenger handling", "#22c55e"),
]

ROUTE_STAT_FEATURES  = [
    "Average journey time", "Number of scheduled trains",
    "Number of cancelled trains", "cancellation_rate",
]
NUMERIC_FEATURES     = ROUTE_STAT_FEATURES + ["year", "month", "day_of_week"]
CATEGORICAL_FEATURES = ["Departure station", "Arrival station", "Service", "season"]

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    "primary":   "#6366f1",   # indigo vif
    "violet":    "#8b5cf6",   # violet
    "sky":       "#0ea5e9",   # bleu ciel
    "teal":      "#14b8a6",   # teal
    "success":   "#22c55e",   # vert vif
    "warning":   "#f59e0b",   # ambre
    "danger":    "#f43f5e",   # rose-rouge vif
    "orange":    "#fb923c",   # orange vif
    "neutral":   "#64748b",
    "bg":        "#f0f4ff",   # fond légèrement bleuté
    "card":      "#ffffff",
    "border":    "#dbeafe",   # bordure bleue légère
    "sidebar":   "#05152e",
    "text":      "#0f172a",
    "muted":     "#475569",   # plus foncé pour meilleur contraste
    "highlight": "#e0e7ff",
    # Chart backgrounds
    "chart_bg":  "rgb(248,250,255)",   # fond chart légèrement bleuté
    "grid":      "#e8eef8",            # grille plus visible
}
# Palette Plotly haute saturation pour les séries
CHART_COLORS = [
    "#6366f1", "#f43f5e", "#22c55e", "#f59e0b",
    "#0ea5e9", "#8b5cf6", "#fb923c", "#14b8a6",
]
# Color scale vivid (remplace RdYlGn_r terne)
RDYLGN  = [[0.0, "#22c55e"], [0.3, "#84cc16"], [0.5, "#f59e0b"],
           [0.75, "#fb923c"], [1.0, "#f43f5e"]]
PLASMA  = "plasma"  # pour les heatmaps

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.main .block-container {{ padding: 1.75rem 2.5rem 3rem; max-width: 1320px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, {C["sidebar"]} 0%, #0d1b3e 55%, #1a1040 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}}
[data-testid="stSidebar"] > div {{ padding: 1.1rem 0.85rem 1.5rem; }}
[data-testid="stSidebar"] * {{ color: #cbd5e1 !important; }}
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stSlider > label {{
    color: #475569 !important; font-size: 0.66rem !important;
    text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700 !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important; border: none !important;
    color: #94a3b8 !important; text-align: left !important;
    padding: 0.5rem 0.75rem !important; border-radius: 10px !important;
    width: 100% !important; font-size: 0.875rem !important;
    font-weight: 500 !important; transition: all 0.15s ease !important;
    margin-bottom: 2px !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(99,102,241,0.18) !important;
    color: #a5b4fc !important; transform: translateX(3px) !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.07) !important; margin: 0.6rem 0 !important;
}}
[data-testid="stSidebar"] .stExpander {{
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important; background: rgba(255,255,255,0.03) !important;
}}
[data-testid="stSidebar"] .stExpander summary {{
    color: #94a3b8 !important; font-size: 0.82rem !important; font-weight: 600 !important;
}}

/* ── Brand ── */
.brand {{ padding: 0.5rem 0.5rem 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 0.5rem; }}
.brand-logo {{
    font-size: 1.7rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1.1;
    background: linear-gradient(130deg, #818cf8 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.brand-tagline {{ font-size: 0.67rem; color: #334155 !important; margin-top: 0.2rem; letter-spacing: 0.03em; }}
.brand-version {{
    display: inline-block; margin-top: 0.4rem; padding: 0.1rem 0.5rem;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 99px; font-size: 0.6rem; color: #818cf8 !important;
    font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
}}

/* ── Sidebar section headers ── */
.sb-section {{
    font-size: 0.6rem; font-weight: 800; color: #334155 !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 0.4rem 0.5rem 0.2rem; margin-top: 0.25rem;
}}

/* ── Filter status pill ── */
.filter-status {{
    background: rgba(79,70,229,0.12); border: 1px solid rgba(79,70,229,0.2);
    border-radius: 8px; padding: 0.5rem 0.75rem; margin-top: 0.5rem; text-align: center;
}}
.filter-status-num {{
    font-size: 1.15rem; font-weight: 800; color: #818cf8 !important; line-height: 1;
}}
.filter-status-lbl {{ font-size: 0.68rem; color: #475569 !important; margin-top: 0.1rem; }}

/* ── Page header ── */
.ph {{ margin-bottom: 1.6rem; }}
.ph-badge {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: {C["highlight"]}; color: {C["primary"]};
    font-size: 0.68rem; font-weight: 800; padding: 0.18rem 0.6rem;
    border-radius: 99px; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.55rem;
}}
.ph-title {{
    font-size: 1.85rem; font-weight: 800; color: {C["text"]};
    margin: 0; letter-spacing: -0.03em; line-height: 1.15;
}}
.ph-sub {{ font-size: 0.88rem; color: {C["muted"]}; margin: 0.35rem 0 0; }}

/* ── KPI cards ── */
.kpi {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 16px; padding: 1.1rem 1.35rem;
    box-shadow: 0 2px 12px rgba(99,102,241,.07);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}}
.kpi:hover {{ box-shadow: 0 8px 28px rgba(99,102,241,.14); transform: translateY(-2px); }}
.kpi::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: var(--kpi-accent, {C["primary"]}); border-radius: 16px 16px 0 0;
}}
.kpi::after {{
    content: ''; position: absolute; bottom: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient(circle, var(--kpi-accent, {C["primary"]}) 0%, transparent 70%);
    opacity: 0.06; border-radius: 50%;
}}
.kpi-icon {{ font-size: 1.5rem; margin-bottom: 0.35rem; }}
.kpi-lbl {{
    font-size: 0.65rem; font-weight: 800; color: {C["muted"]};
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0;
}}
.kpi-val {{
    font-size: 2rem; font-weight: 800; color: var(--kpi-accent, {C["text"]});
    line-height: 1.1; margin: 0.18rem 0 0.1rem; letter-spacing: -0.025em;
}}
.kpi-sub {{ font-size: 0.71rem; color: #94a3b8; margin: 0; }}

/* ── Section title ── */
.stitle {{
    font-size: 0.72rem; font-weight: 800; color: {C["muted"]};
    text-transform: uppercase; letter-spacing: 0.08em;
    margin: 0 0 0.5rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid {C["border"]};
    display: flex; align-items: center; gap: 0.4rem;
}}

/* ── Result card ── */
.result-card {{
    border-radius: 22px; padding: 1.75rem 1.5rem; text-align: center;
    box-shadow: 0 16px 48px rgba(0,0,0,.18);
    position: relative; overflow: hidden;
}}
.result-card::after {{
    content: '🚄'; position: absolute; bottom: -12px; right: -8px;
    font-size: 5rem; opacity: 0.07; pointer-events: none;
}}
.result-route {{
    font-size: 0.8rem; font-weight: 700; opacity: 0.55;
    margin-bottom: 0.5rem; letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.result-val   {{ font-size: 5rem; font-weight: 900; line-height: 1; letter-spacing: -0.05em; }}
.result-unit  {{ font-size: 0.95rem; font-weight: 500; opacity: 0.72; margin-top: 0.1rem; }}
.result-label {{ font-size: 1.25rem; font-weight: 700; margin-top: 0.55rem; }}
.result-msg   {{ font-size: 0.87rem; font-weight: 400; opacity: 0.8; margin-top: 0.3rem; }}

/* ── Step cards (models page) ── */
.step-card {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 16px; padding: 1.4rem; height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,.04); transition: box-shadow 0.2s;
}}
.step-card:hover {{ box-shadow: 0 6px 22px rgba(0,0,0,.08); }}
.step-num {{
    width: 2.1rem; height: 2.1rem; border-radius: 50%;
    background: linear-gradient(135deg, {C["primary"]}, {C["violet"]});
    color: white; font-size: 0.88rem; font-weight: 900;
    display: flex; align-items: center; justify-content: center; margin-bottom: 0.7rem;
}}
.step-title {{ font-size: 0.95rem; font-weight: 700; color: {C["text"]}; margin: 0 0 0.35rem; }}
.step-body  {{ font-size: 0.83rem; color: {C["muted"]}; line-height: 1.6; margin: 0; }}

/* ── Accuracy banner ── */
.acc-banner {{
    border-radius: 16px; padding: 1.35rem 1.75rem; color: white;
    text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,.18);
}}
.acc-banner-val {{ font-size: 2.6rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1.1; }}
.acc-banner-lbl {{ font-size: 0.78rem; opacity: 0.78; margin-top: 0.3rem; }}

/* ── Placeholder ── */
.placeholder {{
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    min-height: 300px; background: linear-gradient(135deg, #f8fafc, #f0f4ff);
    border: 2px dashed {C["border"]}; border-radius: 20px; text-align: center; padding: 2rem;
}}
.placeholder-icon {{ font-size: 3rem; margin-bottom: 0.75rem; opacity: 0.4; }}
.placeholder p {{ color: {C["muted"]}; font-size: 0.93rem; margin: 0; font-weight: 500; max-width: 260px; }}

/* ── Insight box ── */
.insight {{
    background: linear-gradient(135deg, #f0f4ff, #faf5ff);
    border: 1px solid #ddd6fe; border-left: 4px solid {C["primary"]};
    border-radius: 0 12px 12px 0; padding: 0.85rem 1.15rem; margin: 0.5rem 0;
}}
.insight-title {{ font-size: 0.7rem; font-weight: 800; color: {C["primary"]}; margin: 0 0 0.25rem; text-transform: uppercase; letter-spacing: 0.06em; }}
.insight-body  {{ font-size: 0.84rem; color: {C["text"]}; margin: 0; line-height: 1.5; }}

/* ── Separator ── */
.sep {{ height: 1px; background: {C["border"]}; margin: 1.5rem 0; border: none; }}

/* ── Streamlit component overrides ── */
[data-testid="stMetric"] {{
    background: {C["card"]}; border: 1px solid {C["border"]}; border-radius: 12px;
    padding: 0.7rem 1rem;
}}
div[data-testid="stTabs"] button[role="tab"] {{
    font-size: 0.83rem !important; font-weight: 600 !important; padding: 0.5rem 0.85rem !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {C["primary"]}, {C["violet"]}) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4) !important; transition: all 0.2s !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.5) !important;
}}
div[data-testid="stPills"] span {{
    font-size: 0.78rem !important; font-weight: 600 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ── Translations ───────────────────────────────────────────────────────────────
TR = {
    "fr": {
        "lang_btn": "🇬🇧 English",
        "nav_predict": "🎯 Prédire un retard",
        "nav_explore": "📊 Explorer les données",
        "nav_models": "🤖 Les modèles IA",
        "model_title": "Modèle IA",
        "model_help": "Algorithme utilisé pour la prédiction",
        "model_one_only": "Un seul modèle disponible.",
        "filters_title": "Filtres",
        "f_stations": "Gares de départ",
        "f_years": "Période",
        "f_seasons": "Saisons",
        "f_days": "Jours de la semaine",
        "f_service": "Service",
        "f_reset": "🔄 Réinitialiser",
        "f_select_all": "Tout sélectionner",
        "dataset_info": "trajets filtrés",
        "s_winter": "❄️ Hiver", "s_spring": "🌸 Printemps",
        "s_summer": "☀️ Été",   "s_autumn": "🍂 Automne",
        "s_mon":"Lun","s_tue":"Mar","s_wed":"Mer","s_thu":"Jeu",
        "s_fri":"Ven","s_sat":"Sam","s_sun":"Dim",
        # Predict page
        "p_title": "Prédire le retard de ton train",
        "p_sub": "Gare de départ, d'arrivée et date — l'IA prédit ton retard en temps réel.",
        "p_dep": "🚉 Gare de départ",
        "p_arr": "🏁 Gare d'arrivée",
        "p_date": "📅 Date du voyage",
        "p_btn": "🔮 Calculer mon retard estimé",
        "p_approx": "⚠️ Trajet non référencé — estimation via des trajets similaires",
        "p_min": "minutes de retard estimé",
        "p_hist": "Moyenne habituelle",
        "p_sigma": "Imprévisibilité",
        "p_sigma_help": "Plus ce chiffre est grand, plus les retards sont imprévisibles sur ce trajet.",
        "p_by": "Prédiction par **{m}**",
        "p_dist_title": "Historique des retards sur ce trajet",
        "p_pred_marker": "Prédiction : {v:.1f} min",
        "p_avg_marker": "Moyenne",
        "p_compare_title": "Estimation par saison",
        "p_dow_title": "Retard selon le jour du voyage",
        "p_placeholder": "Choisis tes gares et une date, puis clique sur le bouton.",
        # Explore page
        "e_title": "Explorer les données",
        "e_sub": "Toutes les statistiques sur les retards TGV SNCF — 2018 à 2025.",
        "e_k_records": "Trajets analysés",
        "e_k_delay": "Retard moyen",
        "e_k_punct": "À l'heure",
        "e_k_cancel": "Annulés",
        "e_k_delay15": "Retardés >15 min",
        "e_k_delay30": "Retardés >30 min",
        "e_dist": "Distribution des retards",
        "e_trend": "Évolution mensuelle",
        "e_stations": "Top 15 gares les plus en retard",
        "e_heatmap": "Retard moyen : saison × année",
        "e_export": "Exporter les données filtrées",
        "e_dl_btn": "📥 Télécharger CSV",
        "e_avg": "Moy.",
        "e_delay_ax": "Retard (min)",
        "e_count_ax": "Nombre de trajets",
        "e_tab_overview": "📊 Résumé",
        "e_tab_time": "🕐 Temporel",
        "e_tab_routes": "🚂 Trajets",
        "e_tab_causes": "⚡ Causes",
        "e_tab_cancel": "❌ Annulations",
        "e_tab_map": "🗺️ Carte",
        "e_by_dow": "Retard moyen par jour de la semaine",
        "e_by_month": "Retard moyen par mois",
        "e_by_year": "Tendance annuelle (boxplots)",
        "e_top_routes": "Top 20 trajets les plus en retard",
        "e_scatter_jt": "Durée du trajet vs Retard (par trajet)",
        "e_causes_title": "Répartition des causes de retard",
        "e_causes_trend": "Évolution des causes par année (%)",
        "e_cancel_by_st": "Top 15 gares par taux d'annulation",
        "e_cancel_trend": "Évolution du taux d'annulation mensuel",
        "e_map_title": "Carte des retards par gare de départ",
        "e_boxplot_season": "Retard par saison",
        "e_boxplot_year": "Retard par année",
        "e_delay_cat": "Catégories de retard",
        "e_arr_vs_dep": "Retard arrivée vs départ",
        "e_route": "Trajet",
        "e_dep_station": "Gare de départ",
        "e_arr_station": "Gare d'arrivée",
        "e_journey_time": "Durée (min)",
        "e_cancel_rate": "Taux d'annulation",
        # Models page
        "m_title": "Comment ça marche ?",
        "m_sub": "L'IA analyse des millions de trajets réels pour prédire ton retard.",
        "m_step1_title": "Tu choisis ton trajet",
        "m_step1_body": "Deux gares et une date. C'est tout ce dont tu as besoin de fournir.",
        "m_step2_title": "L'IA creuse dans l'historique",
        "m_step2_body": "Elle analyse ce qui s'est passé sur ce trajet : durée moyenne, taux d'annulation, influence des saisons, jour de la semaine...",
        "m_step3_title": "Elle te donne une estimation",
        "m_step3_body": "La prédiction est basée sur des centaines de milliers de trajets réels enregistrés par la SNCF depuis 2018.",
        "m_accuracy_exp": "En moyenne, la prédiction se trompe de **{rmse:.1f} minutes**.",
        "m_imp_title": "Qu'est-ce qui influence le plus le retard ?",
        "m_imp_na": "Information non disponible pour ce modèle.",
        "m_table_title": "Comparaison des modèles IA disponibles",
        "m_rmse": "Erreur moy. (min)",
        "m_r2": "Précision (R²)",
        "m_mae": "MAE (min)",
        "filter_showing": "Affichage :",
        "filter_of": "/",
        "filter_trips": "trajets",
    },
    "en": {
        "lang_btn": "🇫🇷 Français",
        "nav_predict": "🎯 Predict a delay",
        "nav_explore": "📊 Explore data",
        "nav_models": "🤖 AI models",
        "model_title": "AI model",
        "model_help": "Algorithm used for prediction",
        "model_one_only": "Only one model available.",
        "filters_title": "Filters",
        "f_stations": "Departure stations",
        "f_years": "Year range",
        "f_seasons": "Seasons",
        "f_days": "Days of week",
        "f_service": "Service type",
        "f_reset": "🔄 Reset filters",
        "f_select_all": "Select all",
        "dataset_info": "filtered trips",
        "s_winter": "❄️ Winter", "s_spring": "🌸 Spring",
        "s_summer": "☀️ Summer", "s_autumn": "🍂 Autumn",
        "s_mon":"Mon","s_tue":"Tue","s_wed":"Wed","s_thu":"Thu",
        "s_fri":"Fri","s_sat":"Sat","s_sun":"Sun",
        # Predict page
        "p_title": "Predict your train delay",
        "p_sub": "Departure, arrival and date — AI predicts your delay in real time.",
        "p_dep": "🚉 Departure station",
        "p_arr": "🏁 Arrival station",
        "p_date": "📅 Travel date",
        "p_btn": "🔮 Calculate my estimated delay",
        "p_approx": "⚠️ Unknown route — estimate based on similar journeys",
        "p_min": "minutes estimated delay",
        "p_hist": "Usual average",
        "p_sigma": "Unpredictability",
        "p_sigma_help": "Higher = more unpredictable delays on this route.",
        "p_by": "Predicted by **{m}**",
        "p_dist_title": "Historical delays on this route",
        "p_pred_marker": "Prediction: {v:.1f} min",
        "p_avg_marker": "Average",
        "p_compare_title": "Estimate by season",
        "p_dow_title": "Delay by day of travel",
        "p_placeholder": "Pick your stations and a date, then click the button.",
        # Explore page
        "e_title": "Explore data",
        "e_sub": "All statistics about TGV SNCF delays — 2018 to 2025.",
        "e_k_records": "Journeys analysed",
        "e_k_delay": "Avg delay",
        "e_k_punct": "On time",
        "e_k_cancel": "Cancelled",
        "e_k_delay15": "Delayed >15 min",
        "e_k_delay30": "Delayed >30 min",
        "e_dist": "Delay distribution",
        "e_trend": "Monthly trend",
        "e_stations": "Top 15 most delayed stations",
        "e_heatmap": "Avg delay: season × year",
        "e_export": "Export filtered data",
        "e_dl_btn": "📥 Download CSV",
        "e_avg": "Avg",
        "e_delay_ax": "Delay (min)",
        "e_count_ax": "Number of journeys",
        "e_tab_overview": "📊 Overview",
        "e_tab_time": "🕐 Time",
        "e_tab_routes": "🚂 Routes",
        "e_tab_causes": "⚡ Causes",
        "e_tab_cancel": "❌ Cancellations",
        "e_tab_map": "🗺️ Map",
        "e_by_dow": "Avg delay by day of week",
        "e_by_month": "Avg delay by month",
        "e_by_year": "Annual trend (boxplots)",
        "e_top_routes": "Top 20 most delayed routes",
        "e_scatter_jt": "Journey time vs Delay (per route)",
        "e_causes_title": "Delay cause breakdown",
        "e_causes_trend": "Cause trends by year (%)",
        "e_cancel_by_st": "Top 15 stations by cancellation rate",
        "e_cancel_trend": "Monthly cancellation rate trend",
        "e_map_title": "Delay map by departure station",
        "e_boxplot_season": "Delay by season",
        "e_boxplot_year": "Delay by year",
        "e_delay_cat": "Delay categories",
        "e_arr_vs_dep": "Arrival vs departure delay",
        "e_route": "Route",
        "e_dep_station": "Departure station",
        "e_arr_station": "Arrival station",
        "e_journey_time": "Duration (min)",
        "e_cancel_rate": "Cancellation rate",
        # Models page
        "m_title": "How does it work?",
        "m_sub": "The AI analyses millions of real journeys to predict your delay.",
        "m_step1_title": "You choose your journey",
        "m_step1_body": "Two stations and a date. That's all you need to provide.",
        "m_step2_title": "AI digs through history",
        "m_step2_body": "It analyses what happened on this route: average duration, cancellation rate, seasonal influence, day of week...",
        "m_step3_title": "It gives you an estimate",
        "m_step3_body": "The prediction is based on hundreds of thousands of real journeys recorded by SNCF since 2018.",
        "m_accuracy_exp": "On average, the prediction is off by **{rmse:.1f} minutes**.",
        "m_imp_title": "What influences delays the most?",
        "m_imp_na": "Information not available for this model type.",
        "m_table_title": "AI model comparison",
        "m_rmse": "Avg error (min)",
        "m_r2": "Accuracy (R²)",
        "m_mae": "MAE (min)",
        "filter_showing": "Showing:",
        "filter_of": "/",
        "filter_trips": "trips",
    },
}

# ── Session state init ─────────────────────────────────────────────────────────
_defaults = {"lang": "fr", "page": "predict", "prediction": None}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def t(key: str, **kw) -> str:
    txt = TR[st.session_state.lang].get(key, key)
    return txt.format(**kw) if kw else txt


def slabel(sk: str) -> str:
    return t(f"s_{sk}")


def get_delay_info(v: float):
    fr_fn = lambda v: (
        ("A l'heure", "#10b981", "#f0fdf4", "Ton train devrait arriver sans souci.")
        if v < 5 else
        ("Petit retard", "#f59e0b", "#fffbeb", "Un léger retard, reste à portée du quai.")
        if v < 15 else
        ("Retard modéré", "#f97316", "#fff7ed", "Prévois de quoi patienter sur le quai.")
        if v < 30 else
        ("Retard important", "#ef4444", "#fef2f2", "Mieux vaut avoir de la batterie et de quoi lire.")
    )
    en_fn = lambda v: (
        ("On time", "#10b981", "#f0fdf4", "Your train should arrive without issue.")
        if v < 5 else
        ("Slight delay", "#f59e0b", "#fffbeb", "Just a small delay, stay near the platform.")
        if v < 15 else
        ("Moderate delay", "#f97316", "#fff7ed", "Bring something to pass the time.")
        if v < 30 else
        ("Significant delay", "#ef4444", "#fef2f2", "Better have battery and something to read.")
    )
    return (en_fn if st.session_state.lang == "en" else fr_fn)(v)


# ── Geo helpers ───────────────────────────────────────────────────────────────
def _parse_geo(s):
    try:
        lat, lon = str(s).split(", ")
        return float(lat), float(lon)
    except Exception:
        return np.nan, np.nan


def delay_to_rgb(delay: float, d_min: float = 2.5, d_max: float = 11.0,
                 alpha: int = 210) -> list:
    """Convert a delay value to an RGBA list for PyDeck layers."""
    t = float(np.clip((delay - d_min) / (d_max - d_min + 1e-9), 0, 1))
    if t < 0.33:
        r, g, b = int(34 + 220 * t / 0.33), int(197 - 40 * t / 0.33), 50
    elif t < 0.66:
        r, g, b = 255, int(157 - 80 * (t - 0.33) / 0.33), 30
    else:
        r, g, b = 255, int(77 - 60 * (t - 0.66) / 0.34), 20
    return [r, g, b, alpha]


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        "data/processed/trains/cleaned_dataset.csv", parse_dates=["Date"]
    )
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"]       = df["Date"].dt.month
    df["year"]        = df["Date"].dt.year
    df["route"]       = df["Departure station"] + " → " + df["Arrival station"]

    df[["dep_lat", "dep_lon"]] = pd.DataFrame(
        df["departure_station_geo"].map(_parse_geo).tolist(), index=df.index
    )
    df[["arr_lat", "arr_lon"]] = pd.DataFrame(
        df["arrival_station_geo"].map(_parse_geo).tolist(), index=df.index
    )
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


@st.cache_data
def build_station_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station aggregates with geo coords for map view."""
    agg_dict = {
        "dep_lat":          ("dep_lat",  "first"),
        "dep_lon":          ("dep_lon",  "first"),
        "avg_delay":        (TARGET,     "mean"),
        "n_trips":          ("Date",     "count"),
        "cancel_rate":      ("cancellation_rate", "mean"),
        "punct_rate":       ("punctuality_rate",  "mean"),
    }
    # Optional >60min column
    col60 = "Number of trains delayed > 60min"
    if col60 in df.columns:
        agg_dict["n_delay_60"] = (col60, "sum")
    col_sched = "Number of scheduled trains"
    if col_sched in df.columns:
        agg_dict["n_sched"] = (col_sched, "sum")

    result = (
        df.groupby("Departure station")
        .agg(**agg_dict)
        .reset_index()
        .dropna(subset=["dep_lat", "dep_lon"])
        .rename(columns={
            "Departure station": "station",
            "dep_lat": "lat", "dep_lon": "lon",
        })
    )
    # Compute colors and elevations for pydeck
    d_min = result["avg_delay"].min()
    d_max = result["avg_delay"].max()
    result["color"]     = result["avg_delay"].apply(
        lambda d: delay_to_rgb(d, d_min, d_max)
    )
    result["elevation"] = (result["avg_delay"] - d_min) / (d_max - d_min + 1e-9) * 120_000 + 20_000
    result["avg_delay_r"] = result["avg_delay"].round(2)
    result["cancel_pct"]  = (result["cancel_rate"] * 100).round(2)
    result["punct_pct"]   = (result["punct_rate"] * 100).round(1)
    return result


@st.cache_data
def build_route_arcs(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route aggregates with full departure + arrival geo for ArcLayer."""
    col60   = "Number of trains delayed > 60min"
    col_sched = "Number of scheduled trains"
    extra = {}
    if col60    in df.columns: extra["n_delay_60"] = (col60,    "sum")
    if col_sched in df.columns: extra["n_sched"]   = (col_sched, "sum")

    route_agg = (
        df.groupby(["Departure station", "Arrival station"])
        .agg(
            dep_lat     = ("dep_lat", "first"),
            dep_lon     = ("dep_lon", "first"),
            arr_lat     = ("arr_lat", "first"),
            arr_lon     = ("arr_lon", "first"),
            avg_delay   = (TARGET,            "mean"),
            dep_delay   = (DEP_TARGET,        "mean"),
            n_trips     = ("Date",            "count"),
            cancel_rate = ("cancellation_rate","mean"),
            journey_time= ("Average journey time", "mean"),
            **extra,
        )
        .reset_index()
        .dropna(subset=["dep_lat", "dep_lon", "arr_lat", "arr_lon"])
    )
    route_agg["route"] = (route_agg["Departure station"] + " → "
                          + route_agg["Arrival station"])
    d_min = route_agg["avg_delay"].min()
    d_max = route_agg["avg_delay"].max()
    # Source color (departure side): blue tones
    route_agg["src_color"] = route_agg["avg_delay"].apply(
        lambda d: delay_to_rgb(d, d_min, d_max, alpha=180)
    )
    # Target color (arrival side): same but more opaque
    route_agg["tgt_color"] = route_agg["avg_delay"].apply(
        lambda d: delay_to_rgb(d, d_min, d_max, alpha=230)
    )
    # Arc width proportional to trip count (normalised 1–8)
    w_min = route_agg["n_trips"].min()
    w_max = route_agg["n_trips"].max()
    route_agg["arc_width"] = (
        1 + 7 * (route_agg["n_trips"] - w_min) / (w_max - w_min + 1e-9)
    ).round(1)
    route_agg["avg_delay_r"]  = route_agg["avg_delay"].round(2)
    route_agg["cancel_pct"]   = (route_agg["cancel_rate"] * 100).round(2)
    route_agg["journey_time_r"] = route_agg["journey_time"].round(0)
    return route_agg


def discover_models() -> dict:
    meta: dict = {}
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json") as f:
            meta = json.load(f)
    catalog: dict = {}
    for path in sorted(glob.glob("models/*.joblib")):
        try:
            art  = joblib.load(path)
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
            art  = joblib.load("model.joblib")
            name = (
                art.get("model_name", "Modèle") if isinstance(art, dict) else "Modèle"
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
        est     = _pipeline.named_steps["model"]
        cat_enc = (
            _pipeline.named_steps["prep"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_names = cat_enc.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        names     = NUMERIC_FEATURES + cat_names
        imp       = getattr(
            est, "feature_importances_",
            getattr(est, "coef_", np.zeros(len(names))),
        )
        return pd.Series(np.abs(imp), index=names).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(dep, arr, date, pipeline, route_stats):
    row    = route_stats[
        (route_stats["Departure station"] == dep)
        & (route_stats["Arrival station"] == arr)
    ]
    approx = False
    if row.empty:
        row    = route_stats[route_stats["Departure station"] == dep]
        approx = True
    if row.empty:
        row    = route_stats
        approx = True
    if row.empty:
        return None, False
    stats   = row[ROUTE_STAT_FEATURES].mean()
    service = row["Service"].mode().iloc[0]
    m       = date.month
    inp     = pd.DataFrame([{
        "Average journey time":          stats["Average journey time"],
        "Number of scheduled trains":    stats["Number of scheduled trains"],
        "Number of cancelled trains":    stats["Number of cancelled trains"],
        "cancellation_rate":             stats["cancellation_rate"],
        "year":                          date.year,
        "month":                         m,
        "day_of_week":                   date.weekday(),
        "Departure station":             dep,
        "Arrival station":               arr,
        "Service":                       service,
        "season":                        SEASON_MAP[m],
    }])
    return max(0.0, float(pipeline.predict(inp)[0])), approx


# ── Chart helpers ──────────────────────────────────────────────────────────────
_DARK  = C["text"]     # "#0f172a"
_MUTED = C["muted"]    # "#475569"

def chart_style(fig, height: int = 320, margin=None):
    m = margin or dict(t=28, b=24, l=14, r=14)
    axis_common = dict(
        gridcolor=C["grid"], linecolor="#94a3b8",
        zeroline=False, showline=True, linewidth=1,
        tickfont=dict(color=_DARK, size=11, family="Inter"),
        title_font=dict(color=_DARK, size=12, family="Inter"),
    )
    fig.update_layout(
        height=height, margin=m,
        plot_bgcolor=C["chart_bg"], paper_bgcolor="white",
        font=dict(family="Inter", size=12, color=_DARK),
        title_font=dict(color=_DARK),
        legend=dict(font=dict(color=_DARK, size=11)),
        xaxis={**axis_common},
        yaxis={**axis_common},
        coloraxis_colorbar=dict(
            tickfont=dict(color=_DARK, size=10),
            title_font=dict(color=_DARK, size=11),
        ),
    )
    # Thicker lines by default for line/scatter traces
    fig.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(width=2.5),
    )
    return fig


def _hex_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #rrggbb to rgba(r,g,b,a) string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_gauge(v: float, max_val: float = 60) -> go.Figure:
    _, color, _, _ = get_delay_info(v)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(v, 1),
        number={"suffix": " min", "font": {"size": 34, "family": "Inter", "color": C["text"]}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": C["muted"], "tickfont": {"size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5],       "color": "#dcfce7"},
                {"range": [5, 15],      "color": "#fef9c3"},
                {"range": [15, 30],     "color": "#ffedd5"},
                {"range": [30, max_val],"color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": v},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=210, margin=dict(t=8, b=8, l=18, r=18), paper_bgcolor="white")
    return fig


def section_title(label: str, icon: str = ""):
    prefix = f"{icon} " if icon else ""
    st.markdown(f'<p class="stitle">{prefix}{label}</p>', unsafe_allow_html=True)


def kpi_card(col, icon: str, label: str, value: str, sub: str, accent: str = C["primary"]):
    col.markdown(
        f'<div class="kpi" style="--kpi-accent:{accent}">'
        f'<div class="kpi-icon">{icon}</div>'
        f'<p class="kpi-lbl">{label}</p>'
        f'<p class="kpi-val">{value}</p>'
        f'<p class="kpi-sub">{sub}</p></div>',
        unsafe_allow_html=True,
    )


# ── Load data ──────────────────────────────────────────────────────────────────
df          = load_data()
route_stats = build_route_stats(df)
stations    = sorted(df["Departure station"].dropna().unique())
services    = sorted(df["Service"].dropna().unique())
catalog     = discover_models()

ymin, ymax = int(df["year"].min()), int(df["year"].max())

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown(
        '<div class="brand">'
        '<div class="brand-logo">🚄 TARDIS</div>'
        '<div class="brand-tagline">Prédicteur de retards TGV · SNCF</div>'
        '<span class="brand-version">2025 · v2</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Language toggle
    if st.button(t("lang_btn"), key="lang_btn"):
        st.session_state.lang = "en" if st.session_state.lang == "fr" else "fr"
        st.rerun()

    st.divider()

    # Navigation
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    for pid, lbl in [
        ("predict", t("nav_predict")),
        ("explore", t("nav_explore")),
        ("models",  t("nav_models")),
    ]:
        if st.button(lbl, key=f"nav_{pid}"):
            st.session_state.page = pid
            st.rerun()

    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">Modèle IA</div>', unsafe_allow_html=True)

    if not catalog:
        st.warning(t("model_one_only"))
        st.stop()

    model_names = list(catalog.keys())

    def fmt_model(n):
        m = catalog[n]
        return f"{n}  (±{m['RMSE']:.1f} min)" if "RMSE" in m else n

    sel_model = st.selectbox(
        "_", options=model_names, format_func=fmt_model,
        label_visibility="collapsed", help=t("model_help"), key="model_sel",
    )
    pipeline, model_name = load_pipeline(catalog[sel_model]["file"])
    model_meta           = catalog[sel_model]
    importance           = get_importance(pipeline)

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sb-section">Filtres données</div>', unsafe_allow_html=True
    )

    with st.expander("📅 " + t("f_years"), expanded=True):
        year_range = st.slider(
            "_", ymin, ymax, (ymin, ymax), label_visibility="collapsed"
        )

    with st.expander("🚉 " + t("f_stations"), expanded=False):
        col_a, col_b = st.columns(2)
        if col_a.button(t("f_select_all"), key="sel_all_st", use_container_width=True):
            st.session_state["stations_filter"] = list(stations)
        if col_b.button("✕ Aucune", key="sel_none_st", use_container_width=True):
            st.session_state["stations_filter"] = []

        default_stations = st.session_state.get("stations_filter", list(stations))
        sel_stations = st.multiselect(
            "_", stations, default=default_stations,
            label_visibility="collapsed", key="stations_ms",
        )
        st.session_state["stations_filter"] = sel_stations

        sel_services = st.multiselect(
            t("f_service"), services, default=list(services), key="service_ms"
        )

    with st.expander("🗓️ " + t("f_seasons"), expanded=True):
        SEASON_OPTS = {
            t("s_winter"): "winter",
            t("s_spring"): "spring",
            t("s_summer"): "summer",
            t("s_autumn"): "autumn",
        }
        sel_s_labels = st.pills(
            "_", list(SEASON_OPTS.keys()),
            default=list(SEASON_OPTS.keys()),
            selection_mode="multi",
            label_visibility="collapsed",
            key="season_pills",
        )
        sel_seasons = [SEASON_OPTS[s] for s in sel_s_labels] or list(SEASON_OPTS.values())

    with st.expander("📆 " + t("f_days"), expanded=False):
        dow_labels = [t(f"s_{k}") for k in ["mon","tue","wed","thu","fri","sat","sun"]]
        sel_dow_labels = st.pills(
            "_", dow_labels,
            default=dow_labels,
            selection_mode="multi",
            label_visibility="collapsed",
            key="dow_pills",
        )
        sel_dows = [dow_labels.index(l) for l in sel_dow_labels] if sel_dow_labels else list(range(7))

    # Reset button
    if st.button(t("f_reset"), use_container_width=True, key="reset_filters"):
        st.session_state["stations_filter"] = list(stations)
        st.rerun()

    st.divider()

    # Filter status
    _eff_stations = sel_stations or stations
    df_preview = df[
        df["Departure station"].isin(_eff_stations)
        & df["year"].between(*year_range)
        & df["season"].isin(sel_seasons)
        & df["day_of_week"].isin(sel_dows if sel_dows else list(range(7)))
        & df["Service"].isin(sel_services or services)
    ]
    st.markdown(
        f'<div class="filter-status">'
        f'<div class="filter-status-num">{len(df_preview):,}</div>'
        f'<div class="filter-status-lbl">{t("dataset_info")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Apply filters to main dataframe ───────────────────────────────────────────
_eff_stations = sel_stations or stations
_eff_services = sel_services or services
_eff_dows     = sel_dows if sel_dows else list(range(7))

df_f = df[
    df["Departure station"].isin(_eff_stations)
    & df["year"].between(*year_range)
    & df["season"].isin(sel_seasons)
    & df["day_of_week"].isin(_eff_dows)
    & df["Service"].isin(_eff_services)
]

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "predict":
    st.markdown(
        f'<div class="ph">'
        f'<div class="ph-badge">🎯 Prédiction IA</div>'
        f'<p class="ph-title">{t("p_title")}</p>'
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

        if st.button(t("p_btn"), type="primary", use_container_width=True):
            result, approx = predict(dep, arr, date, pipeline, route_stats)
            st.session_state.prediction = {
                "dep": dep, "arr": arr, "date": date,
                "result": result, "approx": approx, "model": model_name,
            }

        st.markdown(
            f'<p style="color:#94a3b8;font-size:0.75rem;margin-top:0.4rem">{t("p_by", m=model_name)}</p>',
            unsafe_allow_html=True,
        )

        # Show some quick stats for selected route
        hist_quick = df[
            (df["Departure station"] == dep) & (df["Arrival station"] == arr)
        ]
        if len(hist_quick) > 0:
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            r1.metric("Retard moyen", f"{hist_quick[TARGET].mean():.1f} min")
            r2.metric("À l'heure", f"{hist_quick['punctuality_rate'].mean()*100:.0f}%")
            r3.metric("Trajets données", f"{len(hist_quick)}")

    with result_col:
        pred_data = st.session_state.get("prediction")

        if not pred_data:
            st.markdown(
                '<div class="placeholder">'
                '<div class="placeholder-icon">🔮</div>'
                f'<p>{t("p_placeholder")}</p></div>',
                unsafe_allow_html=True,
            )
        elif pred_data["result"] is None:
            st.error(t("p_approx"))
        else:
            v      = pred_data["result"]
            dep_   = pred_data["dep"]
            arr_   = pred_data["arr"]
            date_  = pred_data["date"]
            approx_= pred_data.get("approx", False)

            label, color, bg, msg = get_delay_info(v)

            st.markdown(
                f'<div class="result-card" style="background:{bg}; color:{color};">'
                f'<div class="result-route">{dep_} &rarr; {arr_}</div>'
                f'<div class="result-val">{v:.0f}</div>'
                f'<div class="result-unit">{t("p_min")}</div>'
                f'<div class="result-label">{label}</div>'
                f'<div class="result-msg">{msg}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if approx_:
                st.caption(t("p_approx"))

            st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
            st.plotly_chart(make_gauge(v), use_container_width=True)

            hist = df[
                (df["Departure station"] == dep_) & (df["Arrival station"] == arr_)
            ][TARGET].dropna()
            h_mean = hist.mean() if len(hist) else None
            h_std  = hist.std()  if len(hist) else None

            if h_mean is not None:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(t("p_hist"), f"{h_mean:.1f} min",
                           delta=f"{v - h_mean:+.1f} min", delta_color="inverse")
                if h_std is not None:
                    mc2.metric(t("p_sigma"), f"±{h_std:.1f} min", help=t("p_sigma_help"))
                mc3.metric("Min historique", f"{hist.min():.1f} min")

    # ── Second row: charts ──────────────────────────────────────────────────
    pred_data = st.session_state.get("prediction")
    if pred_data and pred_data["result"] is not None:
        v, dep_, arr_, date_ = (
            pred_data["result"], pred_data["dep"],
            pred_data["arr"],   pred_data["date"],
        )
        hist   = df[(df["Departure station"] == dep_) & (df["Arrival station"] == arr_)][TARGET].dropna()
        h_mean = hist.mean() if len(hist) else None

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns(3, gap="medium")

        # Chart 1: Historical distribution
        with ch1:
            section_title(t("p_dist_title"), "📈")
            if len(hist) >= 5:
                _, color, _, _ = get_delay_info(v)
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=hist, nbinsx=30,
                    marker_color=C["primary"], opacity=0.7, name="",
                ))
                fig.add_vline(x=v, line_color=color, line_width=2.5,
                              annotation_text=t("p_pred_marker", v=v),
                              annotation_font_color=color)
                if h_mean:
                    fig.add_vline(x=h_mean, line_color=C["success"], line_dash="dot",
                                  line_width=1.5, annotation_text=t("p_avg_marker"),
                                  annotation_font_color=C["success"])
                fig.update_layout(showlegend=False,
                                  xaxis_title=t("e_delay_ax"),
                                  yaxis_title=t("e_count_ax"))
                st.plotly_chart(chart_style(fig, 280), use_container_width=True)
            else:
                st.info("Pas assez d'historique.")

        # Chart 2: Seasonal comparison
        with ch2:
            section_title(t("p_compare_title"), "🗓️")
            rows = []
            for sk2, m2 in {"winter": 1, "spring": 4, "summer": 7, "autumn": 10}.items():
                try:
                    d2 = date_.replace(month=m2)
                except ValueError:
                    d2 = date_.replace(month=m2, day=28)
                p2, _ = predict(dep_, arr_, d2, pipeline, route_stats)
                if p2 is not None:
                    _, col2, _, _ = get_delay_info(p2)
                    rows.append({
                        "Saison": f"{SEASON_EMOJIS[sk2]} {slabel(sk2)}",
                        t("e_delay_ax"): round(p2, 1), "color": col2,
                    })
            if rows:
                cdf  = pd.DataFrame(rows)
                fig2 = px.bar(cdf, x="Saison", y=t("e_delay_ax"),
                              color="color", color_discrete_map="identity",
                              text=t("e_delay_ax"))
                fig2.update_traces(textposition="outside", textfont_size=12)
                fig2.update_layout(coloraxis_showscale=False, showlegend=False,
                                   xaxis_title="", yaxis_title=t("e_delay_ax"))
                st.plotly_chart(chart_style(fig2, 280), use_container_width=True)

        # Chart 3: Day-of-week comparison
        with ch3:
            section_title(t("p_dow_title"), "📅")
            dow_rows = []
            dow_labels_short = [t(f"s_{k}") for k in ["mon","tue","wed","thu","fri","sat","sun"]]
            for dow_i in range(7):
                try:
                    d3 = date_.replace()
                    # Simulate for each day
                    import copy
                    d3 = datetime.date(date_.year, date_.month,
                                       min(date_.day, 28))
                    # Use the same month but offset day_of_week
                    delta = (dow_i - d3.weekday()) % 7
                    d3 = d3 + datetime.timedelta(days=delta)
                    p3, _ = predict(dep_, arr_, d3, pipeline, route_stats)
                    if p3 is not None:
                        _, c3, _, _ = get_delay_info(p3)
                        dow_rows.append({
                            "Jour": dow_labels_short[dow_i],
                            t("e_delay_ax"): round(p3, 1),
                            "color": c3,
                            "active": dow_i == date_.weekday(),
                        })
                except Exception:
                    pass
            if dow_rows:
                ddf  = pd.DataFrame(dow_rows)
                fig3 = px.bar(ddf, x="Jour", y=t("e_delay_ax"),
                              color="color", color_discrete_map="identity",
                              text=t("e_delay_ax"))
                # Highlight today
                today_idx = date_.weekday()
                fig3.update_traces(textposition="outside", textfont_size=11)
                fig3.update_layout(coloraxis_showscale=False, showlegend=False,
                                   xaxis_title="", yaxis_title=t("e_delay_ax"))
                # Mark selected day with border
                fig3.add_vrect(
                    x0=today_idx - 0.45, x1=today_idx + 0.45,
                    fillcolor="rgba(79,70,229,0.1)", line_color=C["primary"],
                    line_width=2, layer="below",
                    annotation_text="▲", annotation_position="top",
                )
                st.plotly_chart(chart_style(fig3, 280), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "explore":
    st.markdown(
        f'<div class="ph">'
        f'<div class="ph-badge">📊 Exploration</div>'
        f'<p class="ph-title">{t("e_title")}</p>'
        f'<p class="ph-sub">{t("e_sub")}</p></div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        t("e_tab_overview"), t("e_tab_time"), t("e_tab_routes"),
        t("e_tab_causes"),   t("e_tab_cancel"), t("e_tab_map"),
    ])

    # ── TAB 1: Overview ────────────────────────────────────────────────────
    with tabs[0]:
        # 6 KPI cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi_card(k1, "🗂️",  t("e_k_records"), f"{len(df_f):,}",
                 f"{t('filter_showing')} {len(df_f):,} / {len(df):,}", C["primary"])
        kpi_card(k2, "⏱️",  t("e_k_delay"),
                 f"{df_f[TARGET].mean():.1f} min", "retard moyen arrivée", C["orange"])
        kpi_card(k3, "✅",  t("e_k_punct"),
                 f"{df_f['punctuality_rate'].mean()*100:.1f}%", "à l'heure", C["success"])
        kpi_card(k4, "❌",  t("e_k_cancel"),
                 f"{df_f['cancellation_rate'].mean()*100:.2f}%", "trains annulés", C["danger"])

        n15 = (df_f["Number of trains delayed > 15min"].sum() /
               df_f["Number of scheduled trains"].sum() * 100
               if df_f["Number of scheduled trains"].sum() > 0 else 0)
        n30 = (df_f["Number of trains delayed > 30min"].sum() /
               df_f["Number of scheduled trains"].sum() * 100
               if df_f["Number of scheduled trains"].sum() > 0 else 0)
        kpi_card(k5, "⚠️",  t("e_k_delay15"),  f"{n15:.1f}%", ">15 min de retard", C["warning"])
        kpi_card(k6, "🚨",  t("e_k_delay30"),  f"{n30:.1f}%", ">30 min de retard", "#dc2626")

        st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            section_title(t("e_dist"), "📊")
            hist_data = df_f.dropna(subset=[TARGET])
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=hist_data[TARGET], nbinsx=55,
                marker=dict(
                    color=hist_data[TARGET],
                    colorscale=RDYLGN,
                    showscale=False,
                    line=dict(color="white", width=0.4),
                ),
                opacity=0.9,
                name="",
            ))
            fig.update_layout(showlegend=False, bargap=0.04,
                              xaxis_title=t("e_delay_ax"), yaxis_title=t("e_count_ax"))
            st.plotly_chart(chart_style(fig), use_container_width=True)

        with c2:
            section_title(t("e_trend"), "📈")
            mon = df_f.groupby(["year", "month"])[TARGET].mean().reset_index()
            mon["period"] = pd.to_datetime(mon[["year", "month"]].assign(day=1))
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=mon.sort_values("period")["period"],
                y=mon.sort_values("period")[TARGET],
                mode="lines",
                line=dict(color=C["primary"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.15)",
                name="",
            ))
            mv = df_f[TARGET].mean()
            fig2.add_hline(y=mv, line_dash="dot", line_color="#94a3b8", line_width=1.5,
                           annotation_text=f"{t('e_avg')} {mv:.1f} min",
                           annotation_font_color="#64748b",
                           annotation_bgcolor="rgba(255,255,255,0.8)")
            fig2.update_layout(showlegend=False,
                               xaxis_title="", yaxis_title=t("e_delay_ax"))
            st.plotly_chart(chart_style(fig2), use_container_width=True)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            section_title(t("e_stations"), "🏆")
            top = (
                df_f.groupby("Departure station")[TARGET]
                .mean().dropna().sort_values(ascending=False).head(15).reset_index()
            )
            fig3 = px.bar(
                top, x=TARGET, y="Departure station", orientation="h",
                color=TARGET, color_continuous_scale=RDYLGN,
                range_color=[top[TARGET].min() * 0.9, top[TARGET].max() * 1.05],
                labels={TARGET: t("e_delay_ax"), "Departure station": ""},
                text=top[TARGET].round(1),
            )
            fig3.update_layout(
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(title=t("e_delay_ax"), thickness=12, len=0.7),
                yaxis=dict(autorange="reversed"),
                xaxis_title=t("e_delay_ax"),
            )
            fig3.update_traces(
                textposition="outside", textfont_size=10,
                marker_line_color="white", marker_line_width=0.5,
            )
            st.plotly_chart(chart_style(fig3, 430), use_container_width=True)

        with c4:
            section_title(t("e_heatmap"), "🌡️")
            pivot = df_f.pivot_table(values=TARGET, index="year", columns="season", aggfunc="mean")
            pivot = pivot.reindex(columns=["winter", "spring", "summer", "autumn"])
            pivot.columns = [slabel(c) for c in pivot.columns]
            fig4 = px.imshow(
                pivot, text_auto=".1f",
                color_continuous_scale=PLASMA,
                labels={"color": t("e_delay_ax")},
                aspect="auto",
            )
            fig4.update_traces(textfont=dict(size=13, family="Inter", color="white"))
            fig4.update_layout(xaxis_title="", yaxis_title="Année")
            st.plotly_chart(chart_style(fig4, 430), use_container_width=True)

        # Export
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        section_title(t("e_export"), "💾")
        buf = io.BytesIO()
        df_f.to_csv(buf, index=False)
        st.download_button(
            t("e_dl_btn"), data=buf.getvalue(),
            file_name="tardis_export.csv", mime="text/csv",
            use_container_width=True,
        )

    # ── TAB 2: Temporal ────────────────────────────────────────────────────
    with tabs[1]:
        t1c1, t1c2 = st.columns(2, gap="medium")

        with t1c1:
            section_title(t("e_by_dow"), "📅")
            dow_labels_full = [t(f"s_{k}") for k in ["mon","tue","wed","thu","fri","sat","sun"]]
            dow_delay = df_f.groupby("day_of_week")[TARGET].mean().reset_index()
            dow_delay["Jour"] = dow_delay["day_of_week"].map(
                lambda x: dow_labels_full[x] if x < len(dow_labels_full) else x
            )
            dow_delay["is_wknd"] = dow_delay["day_of_week"] >= 5
            fig_dow = go.Figure()
            for _, row in dow_delay.iterrows():
                clr = C["violet"] if row["is_wknd"] else C["primary"]
                fig_dow.add_trace(go.Bar(
                    x=[row["Jour"]], y=[row[TARGET]],
                    text=[f"{row[TARGET]:.1f}"],
                    textposition="outside", textfont_size=12,
                    marker_color=clr,
                    marker_line_color="white", marker_line_width=1.5,
                    showlegend=False, name="",
                ))
            fig_dow.update_layout(barmode="group", xaxis_title="",
                                  yaxis_title=t("e_delay_ax"), showlegend=False)
            st.plotly_chart(chart_style(fig_dow, 320), use_container_width=True)

        with t1c2:
            section_title(t("e_by_month"), "📆")
            month_names_fr = ["Jan","Fév","Mar","Avr","Mai","Jun",
                              "Jul","Aoû","Sep","Oct","Nov","Déc"]
            month_names_en = ["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"]
            m_names = month_names_en if st.session_state.lang == "en" else month_names_fr
            month_delay = df_f.groupby("month")[TARGET].mean().reset_index()
            month_delay["Mois"] = month_delay["month"].map(
                lambda x: m_names[x - 1] if 1 <= x <= 12 else x
            )
            month_delay["color"] = month_delay[TARGET].apply(
                lambda v: delay_to_rgb(v, month_delay[TARGET].min(), month_delay[TARGET].max(), 255)
            )
            fig_mo = go.Figure()
            fig_mo.add_trace(go.Bar(
                x=month_delay["Mois"], y=month_delay[TARGET],
                text=month_delay[TARGET].round(1),
                textposition="outside", textfont_size=11,
                marker_color=month_delay["color"].apply(
                    lambda c: f"rgba({c[0]},{c[1]},{c[2]},0.9)"
                ),
                marker_line_color="white", marker_line_width=1.5,
                name="",
            ))
            fig_mo.update_layout(showlegend=False,
                                  xaxis_title="", yaxis_title=t("e_delay_ax"))
            st.plotly_chart(chart_style(fig_mo, 320), use_container_width=True)

        t2c1, t2c2 = st.columns(2, gap="medium")

        with t2c1:
            section_title(t("e_boxplot_season"), "📦")
            SEASON_CFG = [
                ("winter", slabel("winter"), C["sky"]),
                ("spring", slabel("spring"), "#22c55e"),
                ("summer", slabel("summer"), "#f59e0b"),
                ("autumn", slabel("autumn"), C["orange"]),
            ]
            fig_bs = go.Figure()
            for sk, lbl, clr in SEASON_CFG:
                vals = df_f[df_f["season"] == sk][TARGET].dropna()
                if len(vals) == 0:
                    continue
                fig_bs.add_trace(go.Box(
                    y=vals,
                    name=lbl,
                    # border + median line = solid, saturated color → always visible
                    line=dict(color=clr, width=2.2),
                    # fill = transparent so text/labels behind remain readable
                    fillcolor=_hex_rgba(clr, 0.18),
                    marker=dict(
                        color=clr,
                        size=4,
                        opacity=0.75,
                        line=dict(color="white", width=0.8),
                    ),
                    boxpoints="outliers",
                    showlegend=False,
                ))
            fig_bs.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title=t("e_delay_ax"),
                xaxis=dict(
                    tickfont=dict(color=_DARK, size=12, family="Inter"),
                    title_font=dict(color=_DARK),
                ),
                yaxis=dict(
                    tickfont=dict(color=_DARK, size=11, family="Inter"),
                    title_font=dict(color=_DARK),
                ),
                font=dict(color=_DARK, family="Inter"),
            )
            st.plotly_chart(chart_style(fig_bs, 360), use_container_width=True)

        with t2c2:
            section_title(t("e_boxplot_year"), "📦")
            years_sorted = sorted(df_f["year"].dropna().unique().astype(int))
            fig_by = go.Figure()
            for i, yr in enumerate(years_sorted):
                vals = df_f[df_f["year"] == yr][TARGET].dropna()
                if len(vals) == 0:
                    continue
                clr = CHART_COLORS[i % len(CHART_COLORS)]
                fig_by.add_trace(go.Box(
                    y=vals,
                    name=str(yr),
                    line=dict(color=clr, width=2.2),
                    fillcolor=_hex_rgba(clr, 0.18),
                    marker=dict(
                        color=clr,
                        size=4,
                        opacity=0.75,
                        line=dict(color="white", width=0.8),
                    ),
                    boxpoints="outliers",
                    showlegend=False,
                ))
            fig_by.update_layout(
                showlegend=False,
                xaxis_title="Année",
                yaxis_title=t("e_delay_ax"),
                xaxis=dict(
                    tickfont=dict(color=_DARK, size=12, family="Inter"),
                    title_font=dict(color=_DARK),
                ),
                yaxis=dict(
                    tickfont=dict(color=_DARK, size=11, family="Inter"),
                    title_font=dict(color=_DARK),
                ),
                font=dict(color=_DARK, family="Inter"),
            )
            st.plotly_chart(chart_style(fig_by, 360), use_container_width=True)

        # Delay categories pie
        section_title(t("e_delay_cat"), "🍩")
        cat_map_fr = {"on_time": "À l'heure", "slight": "Léger", "moderate": "Modéré",
                      "severe": "Sévère", "early": "En avance"}
        cat_map_en = {"on_time": "On time", "slight": "Slight", "moderate": "Moderate",
                      "severe": "Severe", "early": "Early"}
        cat_map    = cat_map_en if st.session_state.lang == "en" else cat_map_fr
        cat_colors = {"À l'heure": C["success"], "On time": C["success"],
                      "Léger": C["warning"], "Slight": C["warning"],
                      "Modéré": C["orange"], "Moderate": C["orange"],
                      "Sévère": C["danger"], "Severe": C["danger"],
                      "En avance": C["teal"], "Early": C["teal"]}
        df_cat = df_f["delay_category"].map(cat_map).value_counts().reset_index()
        df_cat.columns = ["Catégorie", "Nombre"]
        fig_cat = px.pie(
            df_cat, names="Catégorie", values="Nombre",
            color="Catégorie", color_discrete_map=cat_colors,
            hole=0.45,
        )
        fig_cat.update_traces(textposition="outside", textinfo="percent+label",
                               textfont_size=12)
        fig_cat.update_layout(showlegend=True, height=340,
                               margin=dict(t=20, b=20, l=20, r=20),
                               paper_bgcolor="white")
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── TAB 3: Routes ──────────────────────────────────────────────────────
    with tabs[2]:
        # Arrival vs Departure delay comparison
        section_title(t("e_arr_vs_dep"), "⬅️➡️")
        arr_dep = df_f.groupby("year").agg(
            arr_delay=(TARGET, "mean"),
            dep_delay=(DEP_TARGET, "mean"),
        ).reset_index()
        fig_ad = go.Figure()
        fig_ad.add_trace(go.Bar(name="Arrivée", x=arr_dep["year"], y=arr_dep["arr_delay"],
                                marker_color=C["primary"]))
        fig_ad.add_trace(go.Bar(name="Départ",  x=arr_dep["year"], y=arr_dep["dep_delay"],
                                marker_color=C["teal"]))
        fig_ad.update_layout(barmode="group", xaxis_title="Année",
                              yaxis_title=t("e_delay_ax"), showlegend=True,
                              xaxis=dict(type="category"))
        st.plotly_chart(chart_style(fig_ad, 300), use_container_width=True)

        r1, r2 = st.columns(2, gap="medium")

        with r1:
            section_title(t("e_top_routes"), "🏆")
            top_routes = (
                df_f.groupby("route")[TARGET]
                .mean().dropna().sort_values(ascending=False).head(20).reset_index()
            )
            fig_tr = px.bar(
                top_routes, x=TARGET, y="route", orientation="h",
                color=TARGET, color_continuous_scale="RdYlGn_r",
                labels={TARGET: t("e_delay_ax"), "route": ""},
                text=top_routes[TARGET].round(1),
            )
            fig_tr.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                xaxis_title=t("e_delay_ax"),
            )
            fig_tr.update_traces(textposition="outside", textfont_size=9)
            st.plotly_chart(chart_style(fig_tr, 540), use_container_width=True)

        with r2:
            section_title(t("e_scatter_jt"), "🔵")
            route_agg = (
                df_f.groupby("route")
                .agg(
                    avg_delay    = (TARGET, "mean"),
                    journey_time = ("Average journey time", "mean"),
                    n            = ("Date", "count"),
                    cancel_rate  = ("cancellation_rate", "mean"),
                )
                .dropna()
                .reset_index()
            )
            fig_sc = px.scatter(
                route_agg.query("n >= 5"),
                x="journey_time", y="avg_delay",
                size="n", color="avg_delay",
                color_continuous_scale=RDYLGN,
                hover_name="route",
                hover_data={"n": True, "cancel_rate": ":.3f",
                            "journey_time": ":.0f", "avg_delay": ":.1f"},
                labels={
                    "avg_delay":    t("e_delay_ax"),
                    "journey_time": t("e_journey_time"),
                    "n":            t("e_count_ax"),
                },
                size_max=28,
            )
            fig_sc.update_traces(
                marker=dict(line=dict(color="white", width=1.5)),
            )
            fig_sc.update_layout(
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(title=t("e_delay_ax"), thickness=12),
                showlegend=False,
            )
            st.plotly_chart(chart_style(fig_sc, 540), use_container_width=True)

        # Route detail table
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        section_title("Détail des trajets (top 30 par retard)", "📋")
        route_table = (
            df_f.groupby(["Departure station", "Arrival station"])
            .agg(
                avg_delay    = (TARGET, "mean"),
                journey_time = ("Average journey time", "mean"),
                n_trips      = ("Date", "count"),
                cancel_rate  = ("cancellation_rate", "mean"),
                punct_rate   = ("punctuality_rate", "mean"),
            )
            .dropna(subset=["avg_delay"])
            .sort_values("avg_delay", ascending=False)
            .head(30)
            .reset_index()
        )
        route_table.columns = [
            t("e_dep_station"), t("e_arr_station"), t("e_delay_ax"),
            t("e_journey_time"), t("e_count_ax"), t("e_cancel_rate"),
            t("e_k_punct"),
        ]
        st.dataframe(
            route_table.style
            .format({
                t("e_delay_ax"):    "{:.1f} min",
                t("e_journey_time"):"{:.0f} min",
                t("e_cancel_rate"): "{:.2%}",
                t("e_k_punct"):     "{:.1%}",
            })
            .background_gradient(subset=[t("e_delay_ax")], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

    # ── TAB 4: Causes ──────────────────────────────────────────────────────
    with tabs[3]:
        lang_key = 1 if st.session_state.lang == "fr" else 2  # 1=fr, 2=en label in CAUSE_COLS
        valid_causes = [(col, fr, en, clr) for col, fr, en, clr in CAUSE_COLS if col in df_f.columns]

        cause_means = {}
        for col, fr_lbl, en_lbl, clr in valid_causes:
            vals = pd.to_numeric(df_f[col], errors="coerce").dropna()
            if len(vals) > 0:
                lbl = fr_lbl if st.session_state.lang == "fr" else en_lbl
                cause_means[lbl] = (vals.mean(), clr)

        ca1, ca2 = st.columns(2, gap="medium")

        with ca1:
            section_title(t("e_causes_title"), "🍩")
            if cause_means:
                labels = list(cause_means.keys())
                values = [v[0] for v in cause_means.values()]
                colors = [v[1] for v in cause_means.values()]
                fig_cause = go.Figure(go.Pie(
                    labels=labels, values=values, hole=0.42,
                    marker=dict(colors=colors, line=dict(color="white", width=2)),
                    textinfo="percent+label", textfont_size=11,
                ))
                fig_cause.update_layout(
                    showlegend=False, height=380,
                    margin=dict(t=20, b=20, l=10, r=10),
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig_cause, use_container_width=True)
            else:
                st.info("Données de causes non disponibles.")

        with ca2:
            section_title(t("e_causes_trend"), "📈")
            trend_rows = []
            for yr in sorted(df_f["year"].unique()):
                yr_df = df_f[df_f["year"] == yr]
                for col, fr_lbl, en_lbl, clr in valid_causes:
                    vals = pd.to_numeric(yr_df[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        lbl = fr_lbl if st.session_state.lang == "fr" else en_lbl
                        trend_rows.append({
                            "Année": yr, "Cause": lbl,
                            "%": vals.mean(), "color": clr,
                        })
            if trend_rows:
                tdf = pd.DataFrame(trend_rows)
                fig_ct = px.line(
                    tdf, x="Année", y="%", color="Cause",
                    color_discrete_map={r["Cause"]: r["color"] for r in trend_rows},
                    markers=True,
                    labels={"Année": "Année", "%": "% du retard"},
                )
                fig_ct.update_layout(
                    xaxis_title="", yaxis_title="% contribution",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(chart_style(fig_ct, 380), use_container_width=True)

        # Stacked bar by year
        section_title("Part de chaque cause par année (barres empilées)", "📊")
        if trend_rows:
            tdf_p = tdf.pivot(index="Année", columns="Cause", values="%").fillna(0)
            color_map_bar = {r["Cause"]: r["color"] for r in trend_rows}
            fig_sb = px.bar(
                tdf_p.reset_index().melt(id_vars="Année"),
                x="Année", y="value", color="Cause",
                color_discrete_map=color_map_bar,
                labels={"value": "% contribution"},
            )
            fig_sb.update_layout(
                xaxis_title="Année", yaxis_title="% contribution",
                xaxis=dict(type="category"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(chart_style(fig_sb, 340), use_container_width=True)

    # ── TAB 5: Cancellations ───────────────────────────────────────────────
    with tabs[4]:
        can1, can2 = st.columns(2, gap="medium")

        with can1:
            section_title(t("e_cancel_trend"), "📈")
            mon_cancel = df_f.groupby(["year", "month"])["cancellation_rate"].mean().reset_index()
            mon_cancel["period"] = pd.to_datetime(mon_cancel[["year", "month"]].assign(day=1))
            fig_ct2 = go.Figure()
            mc_sorted = mon_cancel.sort_values("period")
            fig_ct2.add_trace(go.Scatter(
                x=mc_sorted["period"], y=mc_sorted["cancellation_rate"],
                mode="lines",
                line=dict(color=C["danger"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(244,63,94,0.18)",
                name="",
            ))
            fig_ct2.update_layout(
                showlegend=False,
                xaxis_title="", yaxis_title=t("e_cancel_rate"),
                yaxis_tickformat=".1%",
            )
            st.plotly_chart(chart_style(fig_ct2, 310), use_container_width=True)

        with can2:
            section_title(t("e_cancel_by_st"), "🏆")
            top_cancel = (
                df_f.groupby("Departure station")["cancellation_rate"]
                .mean().dropna().sort_values(ascending=False).head(15).reset_index()
            )
            fig_tc = px.bar(
                top_cancel, x="cancellation_rate", y="Departure station",
                orientation="h", color="cancellation_rate",
                color_continuous_scale="Reds",
                labels={"cancellation_rate": t("e_cancel_rate"), "Departure station": ""},
                text=(top_cancel["cancellation_rate"] * 100).round(2).astype(str) + "%",
            )
            fig_tc.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                xaxis_tickformat=".1%",
            )
            fig_tc.update_traces(textposition="outside", textfont_size=10)
            st.plotly_chart(chart_style(fig_tc, 430), use_container_width=True)

        # Cancellation by season & year heatmap
        section_title("Taux d'annulation : saison × année", "🌡️")
        piv_cancel = df_f.pivot_table(
            values="cancellation_rate", index="year", columns="season", aggfunc="mean"
        )
        piv_cancel = piv_cancel.reindex(columns=["winter", "spring", "summer", "autumn"])
        piv_cancel.columns = [slabel(c) for c in piv_cancel.columns]
        fig_ch = px.imshow(
            piv_cancel * 100, text_auto=".2f",
            color_continuous_scale="Reds",
            labels={"color": "Annulés (%)"},
        )
        fig_ch.update_layout(xaxis_title="", yaxis_title="Année")
        st.plotly_chart(chart_style(fig_ch, 380), use_container_width=True)

    # ── TAB 6: Carte 3D ────────────────────────────────────────────────────
    with tabs[5]:
        station_geo  = build_station_geo(df_f)
        route_arcs   = build_route_arcs(df_f)

        if len(station_geo) == 0:
            st.info("Pas de données géographiques disponibles avec les filtres actuels.")
        else:
            # ── View selector ────────────────────────────────────────────────
            view_mode = st.radio(
                "Vue 3D",
                options=[
                    "🏙️ Colonnes 3D — retard par gare",
                    "🌐 Arcs des trajets — retard par route",
                    "🔀 Vue combinée",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )

            # ── Quick stats ──────────────────────────────────────────────────
            ms1, ms2, ms3, ms4 = st.columns(4)
            worst  = station_geo.nlargest(1, "avg_delay").iloc[0]
            best   = station_geo.nsmallest(1, "avg_delay").iloc[0]
            ms1.metric("🔴 Plus en retard",     worst["station"], f"{worst['avg_delay']:.1f} min")
            ms2.metric("🟢 Plus ponctuelle",    best["station"],  f"{best['avg_delay']:.1f} min")
            ms3.metric("📍 Gares cartographiées", f"{len(station_geo)}")
            ms4.metric("🛤️ Routes représentées",  f"{len(route_arcs)}")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # ── PyDeck layers ────────────────────────────────────────────────
            DARK_MAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"

            # ColumnLayer — 3D pillars per station
            col_layer = pdk.Layer(
                "ColumnLayer",
                data=station_geo,
                get_position=["lon", "lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=22_000,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
                coverage=0.88,
                extruded=True,
            )

            # ScatterplotLayer — halos around stations
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=station_geo,
                get_position=["lon", "lat"],
                get_radius=25_000,
                get_fill_color=[[c[0], c[1], c[2], 40] for c in station_geo["color"]],
                get_line_color=station_geo["color"].tolist(),
                line_width_min_pixels=1,
                stroked=True,
                filled=True,
                pickable=False,
            )

            # ArcLayer — routes connecting stations
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=route_arcs,
                get_source_position=["dep_lon", "dep_lat"],
                get_target_position=["arr_lon", "arr_lat"],
                get_source_color="src_color",
                get_target_color="tgt_color",
                auto_highlight=True,
                width_scale=0.0001,
                get_width="arc_width",
                width_min_pixels=1,
                width_max_pixels=8,
                pickable=True,
                great_circle=False,
            )

            # TextLayer — station name labels
            text_layer = pdk.Layer(
                "TextLayer",
                data=station_geo,
                get_position=["lon", "lat"],
                get_text="station",
                get_size=14,
                get_color=[255, 255, 255, 200],
                get_angle=0,
                get_alignment_baseline="'bottom'",
                get_text_anchor="'middle'",
                pickable=False,
            )

            tooltip_col = {
                "html": (
                    "<b style='color:#a5b4fc'>{station}</b><br/>"
                    "⏱ Retard moyen : <b>{avg_delay_r} min</b><br/>"
                    "✅ Ponctualité : <b>{punct_pct}%</b><br/>"
                    "❌ Annulations : <b>{cancel_pct}%</b><br/>"
                    "🗂 Nb trajets : <b>{n_trips}</b>"
                ),
                "style": {
                    "backgroundColor": "#0f172a",
                    "color": "#e2e8f0",
                    "fontSize": "13px",
                    "padding": "10px 14px",
                    "borderRadius": "8px",
                    "border": "1px solid #334155",
                },
            }
            tooltip_arc = {
                "html": (
                    "<b style='color:#86efac'>{route}</b><br/>"
                    "⏱ Retard moyen : <b>{avg_delay_r} min</b><br/>"
                    "⏩ Retard départ : <b>{dep_delay:.1f} min</b><br/>"
                    "⏱ Durée trajet : <b>{journey_time_r} min</b><br/>"
                    "❌ Annulations : <b>{cancel_pct}%</b><br/>"
                    "🗂 Trajets : <b>{n_trips}</b>"
                ),
                "style": {
                    "backgroundColor": "#0f172a",
                    "color": "#e2e8f0",
                    "fontSize": "13px",
                    "padding": "10px 14px",
                    "borderRadius": "8px",
                    "border": "1px solid #334155",
                },
            }

            if view_mode.startswith("🏙️"):
                layers = [scatter_layer, col_layer, text_layer]
                view_state = pdk.ViewState(
                    latitude=46.6, longitude=2.6,
                    zoom=4.9, pitch=55, bearing=10,
                    min_zoom=3, max_zoom=12,
                )
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=DARK_MAP,
                    tooltip=tooltip_col,
                )
                st.markdown(
                    '<p style="color:#64748b;font-size:0.78rem;margin-bottom:0.4rem">'
                    "🏙️ <b>Colonnes 3D</b> — hauteur = retard moyen · couleur verte→rouge = faible→fort · "
                    "survolez une gare pour les détails</p>",
                    unsafe_allow_html=True,
                )
                st.pydeck_chart(deck, use_container_width=True, height=580)

            elif view_mode.startswith("🌐"):
                layers = [scatter_layer, arc_layer]
                view_state = pdk.ViewState(
                    latitude=46.6, longitude=2.6,
                    zoom=4.9, pitch=45, bearing=-10,
                    min_zoom=3, max_zoom=12,
                )
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=DARK_MAP,
                    tooltip=tooltip_arc,
                )
                st.markdown(
                    '<p style="color:#64748b;font-size:0.78rem;margin-bottom:0.4rem">'
                    "🌐 <b>Arcs des trajets</b> — épaisseur = fréquence · couleur verte→rouge = retard faible→fort · "
                    "survolez un arc pour les détails</p>",
                    unsafe_allow_html=True,
                )
                st.pydeck_chart(deck, use_container_width=True, height=580)

            else:
                # Combined: columns + arcs
                layers = [scatter_layer, arc_layer, col_layer]
                view_state = pdk.ViewState(
                    latitude=46.6, longitude=2.6,
                    zoom=4.8, pitch=50, bearing=5,
                    min_zoom=3, max_zoom=12,
                )
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=DARK_MAP,
                    tooltip=tooltip_col,
                )
                st.markdown(
                    '<p style="color:#64748b;font-size:0.78rem;margin-bottom:0.4rem">'
                    "🔀 <b>Vue combinée</b> — colonnes par gare + arcs des routes · "
                    "survolez les éléments pour les détails</p>",
                    unsafe_allow_html=True,
                )
                st.pydeck_chart(deck, use_container_width=True, height=580)

            # ── Legend ───────────────────────────────────────────────────────
            st.markdown(
                '<div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;'
                'margin-top:0.5rem;padding:0.6rem 1rem;background:#f0f4ff;'
                'border-radius:10px;border:1px solid #dbeafe">'
                '<span style="font-size:0.75rem;font-weight:700;color:#475569">Légende :</span>'
                '<span style="font-size:0.75rem;color:#475569">🟢 Faible retard</span>'
                '<span style="font-size:0.75rem;color:#475569">🟡 Retard modéré</span>'
                '<span style="font-size:0.75rem;color:#475569">🔴 Retard important</span>'
                '<span style="font-size:0.75rem;color:#94a3b8">· '
                f'Données 2018–2025 · {len(route_arcs)} routes · {len(station_geo)} gares</span>'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "models":
    st.markdown(
        f'<div class="ph">'
        f'<div class="ph-badge">🤖 Intelligence Artificielle</div>'
        f'<p class="ph-title">{t("m_title")}</p>'
        f'<p class="ph-sub">{t("m_sub")}</p></div>',
        unsafe_allow_html=True,
    )

    # Step cards
    s1, s2, s3 = st.columns(3, gap="medium")
    for col, num, tk, bk in [
        (s1, "1", "m_step1_title", "m_step1_body"),
        (s2, "2", "m_step2_title", "m_step2_body"),
        (s3, "3", "m_step3_title", "m_step3_body"),
    ]:
        col.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">{num}</div>'
            f'<p class="step-title">{t(tk)}</p>'
            f'<p class="step-body">{t(bk)}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # Accuracy banners
    if "RMSE" in model_meta:
        rmse    = model_meta["RMSE"]
        r2      = model_meta.get("R2", 0)
        mae     = model_meta.get("MAE", 0)
        acc_pct = max(0, min(100, r2 * 100))

        ac1, ac2, ac3, ac4 = st.columns(4)
        ac1.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,{C["primary"]},{C["violet"]})">'
            f'<div class="acc-banner-val">±{rmse:.1f}</div>'
            f'<div class="acc-banner-lbl">{t("m_rmse")}</div></div>',
            unsafe_allow_html=True,
        )
        ac2.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,#059669,#10b981)">'
            f'<div class="acc-banner-val">{acc_pct:.0f}%</div>'
            f'<div class="acc-banner-lbl">{t("m_r2")} (R²)</div></div>',
            unsafe_allow_html=True,
        )
        ac3.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,#0ea5e9,#6366f1)">'
            f'<div class="acc-banner-val">{mae:.1f}</div>'
            f'<div class="acc-banner-lbl">{t("m_mae")}</div></div>',
            unsafe_allow_html=True,
        )
        ac4.markdown(
            f'<div class="acc-banner" style="background:linear-gradient(135deg,#f59e0b,#f97316)">'
            f'<div class="acc-banner-val">{model_name.split()[0]}</div>'
            f'<div class="acc-banner-lbl">Modèle actif</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(t("m_accuracy_exp", rmse=rmse))

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        section_title(t("m_imp_title"), "🔍")
        if not importance.empty:
            FEATURE_LABELS = {
                "Average journey time":          "Durée du trajet" if st.session_state.lang == "fr" else "Journey time",
                "Number of scheduled trains":    "Trains programmés" if st.session_state.lang == "fr" else "Scheduled trains",
                "Number of cancelled trains":    "Trains annulés" if st.session_state.lang == "fr" else "Cancelled trains",
                "cancellation_rate":             "Taux d'annulation" if st.session_state.lang == "fr" else "Cancellation rate",
                "month":                         "Mois" if st.session_state.lang == "fr" else "Month",
                "day_of_week":                   "Jour de la semaine" if st.session_state.lang == "fr" else "Day of week",
                "year":                          "Année" if st.session_state.lang == "fr" else "Year",
            }
            top15  = importance.head(15).sort_values()
            labels = [
                FEATURE_LABELS.get(
                    i,
                    i.replace("_", " ")
                     .replace("Departure station_", "Départ : " if st.session_state.lang == "fr" else "From: ")
                     .replace("Arrival station_",   "Arrivée : " if st.session_state.lang == "fr" else "To: "),
                )
                for i in top15.index
            ]
            colors = [
                C["primary"] if importance[i] > importance.median() else "#a5b4fc"
                for i in top15.index
            ]
            fig_imp = go.Figure(go.Bar(
                x=top15.values, y=labels, orientation="h",
                marker_color=colors,
                text=[f"{v:.3f}" for v in top15.values],
                textposition="outside",
            ))
            fig_imp.update_layout(
                xaxis_title="Importance relative", yaxis_title="", showlegend=False
            )
            st.plotly_chart(chart_style(fig_imp, 500), use_container_width=True)
        else:
            st.info(t("m_imp_na"))

    with right:
        if len(catalog) > 1:
            section_title(t("m_table_title"), "📋")
            rows_m = [
                {
                    "Modèle": name,
                    t("m_rmse"): meta["RMSE"],
                    t("m_r2"):   meta["R2"],
                    t("m_mae"):  meta.get("MAE", np.nan),
                }
                for name, meta in catalog.items()
                if "RMSE" in meta and "R2" in meta
            ]
            if rows_m:
                perf     = pd.DataFrame(rows_m).sort_values(t("m_rmse"))
                best_idx = perf[t("m_rmse")].idxmin()

                def highlight(row):
                    return (
                        ["background-color:#f0fdf4;font-weight:700"] * len(row)
                        if row.name == best_idx
                        else [""] * len(row)
                    )

                st.dataframe(
                    perf.style.apply(highlight, axis=1)
                    .format({
                        t("m_rmse"): "±{:.2f} min",
                        t("m_r2"):   "{:.1%}",
                        t("m_mae"):  "{:.2f} min",
                    })
                    .bar(subset=[t("m_rmse")], color="#fecaca", vmin=0)
                    .bar(subset=[t("m_r2")],   color="#bbf7d0", vmin=0, vmax=1),
                    use_container_width=True,
                    hide_index=True,
                )
                best_row = perf.iloc[0]
                st.success(
                    f"{'Meilleur modèle' if st.session_state.lang == 'fr' else 'Best model'} : "
                    f"**{best_row['Modèle']}** — ±{best_row[t('m_rmse')]:.1f} min"
                )

                # Radar chart comparing all models
                if len(perf) >= 3:
                    section_title("Comparaison visuelle (radar)", "🕸️")
                    metrics_radar = [t("m_rmse"), t("m_r2"), t("m_mae")]
                    fig_radar = go.Figure()
                    radar_colors = [C["primary"], C["teal"], C["success"],
                                    C["warning"], C["danger"], C["orange"], C["violet"]]
                    for i, (_, row) in enumerate(perf.iterrows()):
                        # Normalize: rmse and mae lower=better, r2 higher=better
                        rmse_n = 1 - (row[t("m_rmse")] - perf[t("m_rmse")].min()) / (perf[t("m_rmse")].max() - perf[t("m_rmse")].min() + 1e-9)
                        r2_n   = row[t("m_r2")]
                        mae_n  = 1 - (row[t("m_mae")] - perf[t("m_mae")].min()) / (perf[t("m_mae")].max() - perf[t("m_mae")].min() + 1e-9)
                        vals   = [rmse_n, r2_n, mae_n]
                        cats   = ["Précision RMSE", "R²", "Précision MAE"]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=vals + [vals[0]], theta=cats + [cats[0]],
                            fill="toself", name=row["Modèle"],
                            line_color=radar_colors[i % len(radar_colors)],
                            fillcolor=f"rgba({int(radar_colors[i % len(radar_colors)][1:3], 16)},"
                                      f"{int(radar_colors[i % len(radar_colors)][3:5], 16)},"
                                      f"{int(radar_colors[i % len(radar_colors)][5:7], 16)},0.1)",
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True, height=380,
                        margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor="white",
                        legend=dict(orientation="v", font=dict(size=10)),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
