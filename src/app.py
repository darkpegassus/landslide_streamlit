import pandas as pd
import streamlit as st

from config import NUMERIC_INPUTS
from data import collect_inputs
from model import load_model, predict, top_feature_importance

from ui import (
    inject_cursor_glow, 
    inject_styles, 
    render_current_profile, 
    render_feature_importance, 
    render_history, 
    render_prediction, 
    render_system_information,
    render_interactive_map  
)

st.set_page_config(page_title="Landslide Risk Predictor", page_icon="src/icons/neural.png", layout="wide", initial_sidebar_state="collapsed")
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True
if "history" not in st.session_state: st.session_state.history = []

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
    st.markdown('<div class="section-title">Current Parameters & Location</div>',unsafe_allow_html=True)
    st.caption("Set the location and 11 raw inputs. The map marker updates as you adjust coordinates.")
    
    # MODIFIED: Create two main columns for inputs and map
    input_panel, map_panel = st.columns([1.6, 1], gap="large")
    
    with input_panel:
        values = collect_inputs()
        primary, secondary, _ = st.columns([1.25, .85, 2.4])
        predict_clicked = primary.button("Predict landslide risk", key="predict",use_container_width=True)
        reset_clicked = secondary.button("Reset history",key="reset",use_container_width=True)
    
    with map_panel:
        # NEW: Render the map using Lat/Lon from the values dictionary
        # Note: 'Latitude' and 'Longitude' must be added to data.py as shown in previous step
        render_interactive_map(values.get("Latitude", 20.59), values.get("Longitude", 78.96), dark)

if reset_clicked:
    st.session_state.history = []; st.session_state.pop("latest_probability",None)

if predict_clicked:
    try:
        # 1. REMOVE Lat/Lon before sending to model
        model_inputs = {k: v for k, v in values.items() if k not in ["Latitude", "Longitude"]}
        
        # 2. Call predict with the cleaned inputs
        probability, decision = predict(loaded_model, model_inputs)
        
        st.session_state.latest_probability = probability
        st.session_state.history.append({**values, "Risk_Probability": probability*100})
    except Exception as error: 
        st.error(f"Prediction failed: {error}")

if "latest_probability" in st.session_state:
    render_prediction(st.session_state.latest_probability,loaded_model.threshold,dark)

st.markdown('<div class="section-title" style="margin-top:1rem">Model Insights</div>',unsafe_allow_html=True)
importance_col, info_col = st.columns(2,gap="large")
with importance_col: render_feature_importance(top_feature_importance(loaded_model),dark)
with info_col: render_system_information(loaded_model.metadata,loaded_model.threshold,dark)

if st.session_state.history:
    profile_col, history_col = st.columns(2,gap="large")
    with profile_col: 
        # MODIFIED: Pass only numeric features to the profile chart
        model_numeric_only = {k: v for k, v in values.items() if k in NUMERIC_INPUTS}
        render_current_profile(model_numeric_only, NUMERIC_INPUTS, dark)
    with history_col: render_history(pd.DataFrame(st.session_state.history),loaded_model.threshold,dark)

st.markdown('<div class="footer">Prototype model trained on synthetic data. Results are for demonstration and research purposes and are not a substitute for professional hazard assessment.<br>Built with Streamlit &middot; Model served via joblib</div>',unsafe_allow_html=True)
