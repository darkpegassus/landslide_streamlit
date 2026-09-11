import pandas as pd
import streamlit as st
import numpy as np

from config import (
    INPUT_COLUMNS,
    NUMERIC_INPUTS,
    MONITORING_STATIONS,
    NER_GEOGRAPHIC_REGIONS
)

from data import (
    collect_inputs,
    fetch_live_ner_weather,
    get_24h_telemetry
)

from model import (
    load_model, 
    predict, 
    top_feature_importance, 
    compute_local_shap,         
    compute_factor_of_safety    
)
from ui import (
    theme,
    inject_cursor_glow, 
    inject_styles, 
    render_current_profile, 
    render_history, 
    render_prediction, 
    render_system_information,
    render_interactive_map,
    render_model_insights_dual, 
    render_multilingual_cap_suite
)

st.set_page_config(
    page_title="Landslide Risk Predictor - MDoNER", 
    page_icon="src/icons/neural.png", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 1. State Management
if "dark_mode" not in st.session_state: 
    st.session_state.dark_mode = True

if "selected_station" not in st.session_state: 
    st.session_state.selected_station = "Aizawl"

if "sim_history" not in st.session_state: 
    st.session_state.sim_history = []

if "current_display_prob" not in st.session_state:
    st.session_state.current_display_prob = 0.50

if "latest_live" not in st.session_state:
    st.session_state.latest_live = {}

def toggle_theme() -> None:
    st.session_state.dark_mode = not st.session_state.dark_mode

# Top Theme Bar
top_spacer, theme_card_col = st.columns([10.5, 3])
with theme_card_col:
    theme_label = "Light mode" if st.session_state.dark_mode else "Dark mode"
    theme_icon = "\u2600" if st.session_state.dark_mode else "\u263e"
    st.button(
        f"{theme_icon}  {theme_label}",
        key="theme-toggle-container",
        on_click=toggle_theme,
        use_container_width=True,
    )

dark = st.session_state.dark_mode
inject_styles(dark)
inject_cursor_glow()

st.markdown(
    '<div class="hero"><h1>GeoSentinel - Landslide Risk Assessment System</h1>'
    '<p>AI-Powered Early Warning & Continuous Telemetry System for the North Eastern Region.</p></div>',
    unsafe_allow_html=True
)

# Load Scikit-Learn Model
try: 
    loaded_model = load_model()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

# --- Main Container: Controls & GIS Map ---
with st.container(border=True, key="current-parameters"):
    st.markdown('<div class="section-title">Monitoring & Simulation Controls</div>', unsafe_allow_html=True)
    st.caption(f"📡 Real-Time Satellite Telemetry Active &middot; Monitoring Station: **{st.session_state.selected_station}**")

    # 1. Fetch live 24h telemetry (always uses Open-Meteo live satellite feeds)
    live_df_24h = get_24h_telemetry(st.session_state.selected_station)
    latest_live = live_df_24h.iloc[-1].to_dict()
    st.session_state.latest_live = latest_live

    # SPLIT: Sliders (Left) vs GIS Map (Right)
    in_col, map_col = st.columns([1.6, 1], gap="large")

    with map_col:
        clicked_site = render_interactive_map(
            st.session_state.selected_station, 
            dark, 
            current_risk_prob=st.session_state.current_display_prob
        )
        if clicked_site != st.session_state.selected_station:
            st.session_state.selected_station = clicked_site
            
            # Immediately synchronize slider keys in session state with live data
            new_telemetry = get_24h_telemetry(clicked_site)
            new_latest = new_telemetry.iloc[-1].to_dict()
            for feat, val in new_latest.items():
                k = "input_lat" if feat == "Latitude" else ("input_lon" if feat == "Longitude" else f"input_{feat}")
                st.session_state[k] = float(val) if isinstance(val, (int, float)) else val
                
            # Recalculate prediction gauge for the newly clicked station immediately
            clean_new = {k: v for k, v in new_latest.items() if k in INPUT_COLUMNS}
            p, _ = predict(loaded_model, clean_new)
            st.session_state.current_display_prob = p
            st.rerun()

    with in_col:
        values = collect_inputs(defaults=latest_live)
        values["Effective_Rainfall_mm"] = float(values.get("Rainfall_3Day", 0) * values.get("Vegetation_Cover", 0.5))

        # Robust simulation check with tolerance to prevent rounding false-positives
        is_sim = any(
            abs(float(values[k]) - float(latest_live[k])) > 1.0 
            for k in ["Slope_Angle", "Rainfall_3Day", "Elevation_m", "Soil_Erosion_Rate"]
            if k in values and k in latest_live
        ) or any(
            abs(float(values[k]) - float(latest_live[k])) > 0.05 
            for k in ["Soil_Saturation", "Vegetation_Cover", "Pore_Pressure_Ratio", "Seismic_PGA_g"]
            if k in values and k in latest_live
        )

        btn_c1, btn_c2, _ = st.columns([1.2, 0.9, 2])
        predict_clicked = btn_c1.button("Predict landslide risk", key="predict", use_container_width=True)
        reset_clicked = btn_c2.button("Reset to Live", key="reset_live", use_container_width=True)

        if reset_clicked:
            for feat, val in latest_live.items():
                k = "input_lat" if feat == "Latitude" else ("input_lon" if feat == "Longitude" else f"input_{feat}")
                st.session_state[k] = float(val) if isinstance(val, (int, float)) else val
            clean_live = {k: v for k, v in latest_live.items() if k in INPUT_COLUMNS}
            p, _ = predict(loaded_model, clean_live)
            st.session_state.current_display_prob = p
            st.session_state.sim_history = []
            st.rerun()

# Prediction Execution Logic
if predict_clicked:
    # Run user manual scenario
    clean_inputs = {k: v for k, v in values.items() if k in INPUT_COLUMNS}
    prob, _ = predict(loaded_model, clean_inputs)
    st.session_state.current_display_prob = prob
    st.session_state.sim_history.append({**values, "Risk_Probability": prob * 100})
elif not is_sim:
    # Continuously sync gauge with live telemetry
    clean_live = {k: v for k, v in latest_live.items() if k in INPUT_COLUMNS}
    live_p, _ = predict(loaded_model, clean_live)
    st.session_state.current_display_prob = live_p

# --- Status Banner ---
status_color = "#f0954f" if dark else "#634ac3"
text_style = (
    f"color: {theme(dark)['text']}; font-weight: 600; padding: 10px; "
    f"border-left: 5px solid {status_color}; background: rgba(0,0,0,0.08); "
    f"border-radius: 5px; margin-bottom: 15px;"
)

if is_sim:
    st.markdown(
        f'<div style="{text_style}">⚠️ Simulation Mode: Monitoring modified parameters for {st.session_state.selected_station}</div>', 
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f'<div style="{text_style}">📡 Live Feed: {st.session_state.selected_station} (Continuous Open-Meteo Satellite Sync Active)</div>', 
        unsafe_allow_html=True
    )

# Render main prediction gauge
render_prediction(
    st.session_state.current_display_prob, 
    loaded_model.threshold, 
    dark, 
    active_station=st.session_state.selected_station
)
render_multilingual_cap_suite(
    st.session_state.selected_station,
    st.session_state.current_display_prob,
    dark
)

# --- Model Insights ---
st.markdown('<div class="section-title" style="margin-top:1rem">Model Insights & Geotechnical Diagnostics</div>', unsafe_allow_html=True)
importance_col, info_col = st.columns(2, gap="large")

with importance_col:
    # Compute local SHAP and Factor of Safety for the active input profile
    local_shap = compute_local_shap(loaded_model, values)
    fos_results = compute_factor_of_safety(values)
    
    render_model_insights_dual(
        local_shap, 
        fos_results, 
        st.session_state.current_display_prob, 
        dark
    )

with info_col:
    render_system_information(loaded_model.metadata, loaded_model.threshold, dark)

# --- Dual History Analytics ---
hist_l, hist_r = st.columns(2, gap="large")

with hist_l:
    # Calculate model probabilities across all 144 time points in the 24-hour window
    history_probs = []
    for _, row in live_df_24h.iterrows():
        row_inputs = {k: v for k, v in row.items() if k in INPUT_COLUMNS}
        p, _ = predict(loaded_model, row_inputs)
        history_probs.append(p * 100)
        
    live_df_24h["Risk_Probability"] = history_probs
    
    st.markdown('<div class="section-title">Live Station Trend (Last 24 Hours · 10m Interval)</div>', unsafe_allow_html=True)
    render_history(live_df_24h, loaded_model.threshold, dark, key="live-history-trend")

with hist_r:
    st.markdown('<div class="section-title">User Simulation History</div>', unsafe_allow_html=True)
    if st.session_state.sim_history:
        render_history(pd.DataFrame(st.session_state.sim_history), loaded_model.threshold, dark, key="user-sim-history")
    else:
        st.info("Adjust any slider above and click Predict to record simulated scenarios.")

st.markdown(
    '<div class="footer">MDoNER Landslide Risk Intelligence Platform &middot; '
    'Real-time Satellite & Geotechnical Decision Support System</div>',
    unsafe_allow_html=True
)