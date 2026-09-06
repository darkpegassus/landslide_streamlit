import pandas as pd
import streamlit as st
import numpy as np
from config import MONITORING_STATIONS, INPUT_COLUMNS, CATEGORICAL_OPTIONS, NUMERIC_INPUTS
from data import collect_inputs
from model import load_model, predict, top_feature_importance
from ui import (
    theme,
    inject_cursor_glow, 
    inject_styles, 
    render_current_profile, 
    render_feature_importance, 
    render_history, 
    render_prediction, 
    render_system_information,
    render_interactive_map  # <--- New import
)

st.set_page_config(page_title="Landslide Risk Predictor", page_icon="src/icons/neural.png", layout="wide", initial_sidebar_state="collapsed")
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True
if "selected_station" not in st.session_state: st.session_state.selected_station = "Kedarnath, UT"
if "sim_history" not in st.session_state: st.session_state.sim_history = []
if "history" not in st.session_state: st.session_state.history = []


def get_station_data(name):
    base = MONITORING_STATIONS[name]["base"]
    rows = []
    for i in range(6):
        row = base.copy()
        # Add temporal noise/trend
        trend = (i * 0.02) if MONITORING_STATIONS[name]["risk"] in ["Extreme", "High"] else 0
        
        for k, v in row.items():
            # Check if it's a numeric feature defined in our config
            if k in NUMERIC_INPUTS:
                low_limit, high_limit = NUMERIC_INPUTS[k][0], NUMERIC_INPUTS[k][1]
                
                # Apply noise and trend
                raw_val = float(v * (1.0 + trend + np.random.normal(0, 0.02)))
                
                # CRITICAL: Clip the value so it never exceeds the slider's max/min
                row[k] = float(np.clip(raw_val, low_limit, high_limit))
            
            # Special handling for coordinates (Latitude/Longitude)
            elif k == "Latitude":
                row[k] = float(np.clip(v, 6.0, 38.0))
            elif k == "Longitude":
                row[k] = float(np.clip(v, 68.0, 98.0))
            else:
                # For categorical strings, just keep as is
                row[k] = v
            # After generating Rainfall and Vegetation, update Effective Rainfall
            row["Effective_Rainfall_mm"] = float(row["Rainfall_3Day"] * row["Vegetation_Cover"])
            # Ensure it stays within range limits if necessary
            row["Effective_Rainfall_mm"] = float(np.clip(row["Effective_Rainfall_mm"], 0, 500))    

        rows.append(row)
    return pd.DataFrame(rows)

def toggle_theme() -> None:
    st.session_state.dark_mode = not st.session_state.dark_mode

top_spacer, theme_card_col = st.columns([10.5, 3])
with theme_card_col:
    theme_label = "Light mode" if st.session_state.dark_mode else "Dark mode"
    theme_icon = "\u2600" if st.session_state.dark_mode else "\u263e"
    st.button(
        f"{theme_icon}  {theme_label}",
        key="theme-toggle-container",
        on_click=toggle_theme,
        help=f" {theme_label}.",
        use_container_width=True,
    )
dark = st.session_state.dark_mode
inject_styles(dark)
inject_cursor_glow()
st.markdown('<div class="hero"><h1>Landslide Risk Prediction Dashboard</h1><p>Estimate landslide probability from site, terrain, hydrology, and seismic conditions.</p></div>',unsafe_allow_html=True)

try: loaded_model = load_model()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}"); st.stop()

with st.container(border=True, key="current-parameters"):
    st.markdown('<div class="section-title">Monitoring & Simulation</div>', unsafe_allow_html=True)
    in_col, map_col = st.columns([1.6, 1], gap="large")
    
    # 1. Fetch Live Data for the selected station
    live_df = get_station_data(st.session_state.selected_station)
    latest_live = live_df.iloc[-1].to_dict()

    with map_col:
        # Render map and catch selection
        new_site = render_interactive_map(st.session_state.selected_station, dark)
        # Check if the station actually changed
        if new_site != st.session_state.selected_station:
            # 1. Update the station name
            st.session_state.selected_station = new_site
            # 2. Generate the fresh live data for the NEW station
            new_live_df = get_station_data(new_site)
            new_latest_live = new_live_df.iloc[-1].to_dict()
            # 3. CRITICAL: Manually update the widget session state keys
            # This forces the sliders to jump to the new values
            for feature, value in new_latest_live.items():
                # Handle the specific keys used in data.py
                if feature == "Latitude":
                    key = "input_lat"
                elif feature == "Longitude":
                    key = "input_lon"
                else:
                    key = f"input_{feature}"
                # Force the value into session state (cast numbers to float for sliders)
                if isinstance(value, (int, float)):
                    st.session_state[key] = float(value)
                else:
                    st.session_state[key] = value
            # 4. Rerun the app to reflect the new slider positions
            st.rerun()

    with in_col:
        st.caption(f"Station: {st.session_state.selected_station} | Status: {MONITORING_STATIONS[st.session_state.selected_station]['risk']} Risk")
        # 2. Sliders now accept live data as defaults
        values = collect_inputs(defaults=latest_live)
        values["Effective_Rainfall_mm"] = float(values["Rainfall_3Day"] * values["Vegetation_Cover"])
        # 3. Check if user changed anything
        is_sim = any(values[k] != latest_live[k] for k in latest_live if k in values)
        
        primary, secondary, _ = st.columns([1.25, .85, 2.4])
        predict_clicked = primary.button("Predict landslide risk", key="predict", use_container_width=True)
        reset_clicked = secondary.button("Reset Simulation", key="reset", use_container_width=True)

if reset_clicked:
    st.session_state.sim_history = []
    st.rerun()

model_inputs = {k: v for k, v in values.items() if k not in ["Latitude", "Longitude"]}
# Initialization of prediction state if not present
if "current_display_prob" not in st.session_state:
    # Default to live data prediction on first load
    p, _ = predict(loaded_model, latest_live)
    st.session_state.current_display_prob = p

# Update ONLY when button is clicked
if predict_clicked:
    probability, decision = predict(loaded_model, model_inputs)
    st.session_state.current_display_prob = probability # Update the display
    
    if is_sim:
        # Record to simulation history
        st.session_state.sim_history.append({**values, "Risk_Probability": probability*100})
    st.rerun()

# Instead of st.warning/st.success, use this theme-aware HTML block:
status_color = "#f0954f" if dark else "#634ac3" # Use accent colors
text_style = f"color: {theme(dark)['text']}; font-weight: 600; padding: 10px; border-left: 5px solid {status_color}; background: rgba(0,0,0,0.1); border-radius: 5px; margin-bottom: 15px;"

if is_sim:
    st.markdown(f'<div style="{text_style}">⚠️ Simulation Mode: Monitoring modified parameters for {st.session_state.selected_station}</div>', unsafe_allow_html=True)
else:
    # Auto-update display prob if site just changed and we are in Live mode
    p_live, _ = predict(loaded_model, latest_live)
    st.session_state.current_display_prob = p_live
    st.markdown(f'<div style="{text_style}">📡 Live Feed: {st.session_state.selected_station}</div>', unsafe_allow_html=True)

render_prediction(st.session_state.current_display_prob, loaded_model.threshold, dark)

st.markdown('<div class="section-title" style="margin-top:1rem">Model Insights</div>',unsafe_allow_html=True)
importance_col, info_col = st.columns(2,gap="large")
with importance_col: render_feature_importance(top_feature_importance(loaded_model),dark)
with info_col: render_system_information(loaded_model.metadata,loaded_model.threshold,dark)

# Create two columns for Live History vs User Manual History
hist_l, hist_r = st.columns(2, gap="large")

with hist_l:
    # Calculate Risk for the 6 historical rows (Live Trend)
    live_probs = []
    for _, r in live_df.iterrows():
        p, _ = predict(loaded_model, {k:v for k,v in r.items() if k in INPUT_COLUMNS})
        live_probs.append(p * 100)
    live_df["Risk_Probability"] = live_probs
    
    st.markdown('<div class="section-title">Live Station Trend (Last 60m)</div>', unsafe_allow_html=True)
    render_history(live_df, loaded_model.threshold, dark, key="live-history-trend")

with hist_r:
    st.markdown('<div class="section-title">User Simulation History</div>', unsafe_allow_html=True)
    if st.session_state.sim_history:
        render_history(pd.DataFrame(st.session_state.sim_history), loaded_model.threshold, dark, key="user-sim-history")
    else:
        st.info("Adjust sliders and click Predict to record simulated scenarios.")

st.markdown('<div class="footer">Prototype model trained on synthetic data. Results are for demonstration and research purposes and are not a substitute for professional hazard assessment.<br>Built with Streamlit &middot; Model served via joblib</div>',unsafe_allow_html=True)
