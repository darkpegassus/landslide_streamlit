import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "landslide_model4.pkl"
SCALER_PATH = "landslide_scaler4.pkl"

INPUT_COLUMNS = ["Rainfall_mm", "Slope_Angle", "Soil_Saturation", "Vegetation_Cover"]

# (min, max, default, step, unit, display_format) — synthetic training-data ranges,
# also used for sliders and radar normalization.
PARAM_RANGES = {
    "Rainfall_mm":      (0.0,  300.0, 100.0, 5.0,  "mm", "%.0f"),
    "Slope_Angle":      (5.0,  80.0,  30.0,  1.0,  "°",  "%.0f"),
    "Soil_Saturation":  (0.10, 1.00,  0.50,  0.01, "",   "%.2f"),
    "Vegetation_Cover": (0.00, 1.00,  0.50,  0.01, "",   "%.2f"),
}

st.set_page_config(
    page_title="Landslide Risk Predictor",
    page_icon="src/icons/neural.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme state lives entirely in session_state now — no widget owns it,
# so we control it fully (needed for a hand-built toggle).
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

is_dark_mode = st.session_state.dark_mode

plot_text_color = "#f3e9df" if is_dark_mode else "#403758"
plot_label_color = "#f0dfd3" if is_dark_mode else "#534b70"
plot_grid_color = "rgba(255, 190, 139, 0.20)" if is_dark_mode else "rgba(112, 96, 157, 0.18)"
plot_background = "rgba(18, 13, 11, 0.55)" if is_dark_mode else "rgba(235, 229, 250, 0.82)"
plot_accent_color = "#f0954f" if is_dark_mode else "#8d7ddd"
plot_fill_color = "rgba(233, 122, 47, 0.24)" if is_dark_mode else "rgba(193, 178, 255, 0.20)"
plot_marker_color = "#ffc08d" if is_dark_mode else "#f18da2"

# ============================================================
# STYLING — premium dark mountain dashboard theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', Georgia, serif;
    font-size: 14px;
}

/* Premium dark background with layered mountain gradients */
.stApp {
    background:
        radial-gradient(ellipse 58% 29% at 12% 48%, rgba(255, 181, 194, 0.78), transparent 72%),
        radial-gradient(ellipse 52% 34% at 50% 50%, rgba(180, 170, 250, 0.70), transparent 74%),
        radial-gradient(ellipse 50% 38% at 83% 52%, rgba(163, 235, 231, 0.72), transparent 76%),
        radial-gradient(ellipse 70% 42% at 48% 58%, rgba(255, 244, 234, 0.84), transparent 74%),
        linear-gradient(180deg, #fffdfb 0%, #faf8ff 44%, #f4fcfc 100%);
    color: #37344f;
    position: relative;
    overflow: hidden;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background: radial-gradient(
        circle 15rem at var(--cursor-x, 50%) var(--cursor-y, 30%),
        rgba(255, 255, 255, 0.92) 0%,
        rgba(255, 255, 255, 0.54) 28%,
        rgba(255, 255, 255, 0) 70%
    );
    transition: background 0.12s ease-out;
}
.stApp > div {
    position: relative;
    z-index: 1;
}

/* Sidebar — premium charcoal panel */
section[data-testid="stSidebar"] {
    background: rgba(14, 18, 22, 0.82);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border-right: 1px solid rgba(196, 174, 126, 0.22);
    box-shadow: 12px 0 30px rgba(0, 0, 0, 0.25);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f4efe7;
    letter-spacing: 0.5px;
    font-weight: 700;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p {
    color: #c7d0cf !important;
}

/* Premium slider accent */
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #c7a66a !important;
    border-color: #f0d9a6 !important;
}
div[data-testid="stSlider"] .st-emotion-cache-1inwz65,
div[data-testid="stTickBar"] {
    background: linear-gradient(90deg, #1f2c31, #7ca7a1, #d4b579) !important;
}

/* Hero header — luxury card */
.hero {
    padding: 2rem 2.2rem;
    border-radius: 34px;
    background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,250,253,0.58));
    border: 1px solid rgba(255, 255, 255, 0.9);
    box-shadow: 0 20px 42px rgba(126, 109, 174, 0.17), inset 0 1px 0 rgba(255,255,255,0.9);
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    font-weight: 700;
    font-size: 3rem;
    letter-spacing: 0.01em;
    margin: 0;
    line-height: 0.96;
    color: #403758;
}
.hero p {
    color: #625c7d;
    margin-top: 0.55rem;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Refined stat cards */
.param-card {
    background: linear-gradient(180deg, rgba(21, 28, 33, 0.82), rgba(17, 24, 29, 0.9));
    border: 1px solid rgba(196, 174, 126, 0.2);
    border-radius: 24px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 18px 30px rgba(0, 0, 0, 0.22);
    transition: all 0.2s ease;
}
.param-card:hover {
    border-color: rgba(196, 174, 126, 0.45);
    transform: translateY(-2px);
    box-shadow: 0 22px 36px rgba(0, 0, 0, 0.28);
}
.param-card .label {
    font-size: 0.72rem;
    color: #c2b28a;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}
.param-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f5f3ef;
    letter-spacing: -0.02em;
}

/* Risk result card */
.risk-card {
    border-radius: 20px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.85);
    margin-top: 0.5rem;
    max-width: 400px;
    max-height: 400px;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 16px 28px rgba(125, 110, 165, 0.16), inset 0 1px 0 rgba(255,255,255,0.78);
}
.risk-card h2 {
    margin: 0;
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    font-size: 2.9rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1;
}
.risk-card p {
    color: #534b70;
    margin-top: 0.45rem;
    font-size: 0.95rem;
}
.risk-low    { background: linear-gradient(135deg, rgba(181, 241, 222, 0.80), rgba(255,255,255,0.60)); border-color: rgba(112, 202, 169, 0.60); }
.risk-low h2 { color: #37886d; }
.risk-medium { background: linear-gradient(135deg, rgba(255, 228, 174, 0.80), rgba(255,255,255,0.60)); border-color: rgba(232, 182, 99, 0.62); }
.risk-medium h2 { color: #a56d28; }
.risk-high   { background: linear-gradient(135deg, rgba(255, 193, 204, 0.82), rgba(255,255,255,0.60)); border-color: rgba(239, 132, 153, 0.68); }
.risk-high h2 { color: #bd526c; }

/* Premium action buttons */
.stButton>button {
    background: linear-gradient(135deg, #ffb9c4, #c1b2ff 54%, #9de4df);
    color: #403758;
    border: none;
    border-radius: 100px;
    padding: 0.7rem 1.25rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    box-shadow: 0 12px 22px rgba(133, 108, 172, 0.24), inset 0 1px 0 rgba(255,255,255,0.72);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #ffc7cf, #d1c6ff 54%, #b6f1eb);
    transform: translateY(-2px);
    box-shadow: 0 18px 30px rgba(133, 108, 172, 0.32);
    color: #403758;
}

/* ============================================================
   FULLY CUSTOM PILL TOGGLE
   Not built on st.toggle at all. An invisible st.button (opacity 0,
   stretched to fill the pill) captures the click; the pill itself —
   track, both static end icons, and the sliding colored knob — is a
   single hand-drawn HTML block whose colors/position come straight
   from Python state, so there is no fighting Streamlit's internal
   widget markup.
   ============================================================ */
div[class*="st-key-theme-control"] {
    position: relative;
    width: 10.5rem;
    height: 2.6rem;
    cursor: pointer;
}
div[class*="st-key-theme-control"] div[data-testid="stMarkdown"] {
    position: absolute;
    inset: 0.15rem 0.35rem;
    pointer-events: none;
}
div[class*="st-key-theme-toggle-container"] {
    position: absolute;
    inset: 0.15rem 0.35rem;
}
div[class*="st-key-theme-toggle-container"] button {
    width: 100%;
    height: 100%;
    min-height: 0;
    padding: 0;
    margin: 0;
    border: none;
    border-radius: 999px;
    background: transparent !important;
    box-shadow: none !important;
    opacity: 0;
    cursor: pointer;
}
div[class*="st-key-theme-toggle-container"] button:hover,
div[class*="st-key-theme-toggle-container"] button:active,
div[class*="st-key-theme-toggle-container"] button:focus {
    transform: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
.pill-track {
    position: absolute;
    inset: 0;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0.55rem;
    border: 1.5px solid;
    box-shadow: inset 0 1px 3px rgba(43, 52, 66, 0.20);
    transition: background 0.25s ease, border-color 0.25s ease;
}
.pill-text {
    position: relative;
    z-index: 1;
    width: calc(100% - 2rem);
    color: #423a60;
    font-family: 'Poppins', Georgia, serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    line-height: 1;
    text-align: center;
    text-transform: uppercase;
    transition: color 0.25s ease;
}
.pill-static-icon {
    font-size: 0.82rem;
    line-height: 1;
    opacity: 0.55;
    position: relative;
    z-index: 1;
}
.pill-knob {
    position: absolute;
    top: 0.2rem;
    width: 1.85rem;
    height: 1.85rem;
    border-radius: 50%;
    border: 2px solid;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    line-height: 1;
    z-index: 2;
    transition: left 0.25s ease, background 0.25s ease, color 0.25s ease, border-color 0.25s ease;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.30);
}

div[class*="st-key-theme-control"] {
    border: 2px solid #241b42 !important;
    border-radius: 16px;
    padding: 0.15rem 0.35rem;
}
.stApp.dark-mode .pill-text { color: #fff1e5; }
.stApp.dark-mode div[class*="st-key-theme-control"] {
    border-color: #f0954f !important;
}

/* Section headers — warm metallic accent bar */
/* Theme switch: a single native button keeps its icon and text aligned. */
div[class*="st-key-theme-toggle-container"] {
    position: static;
    inset: auto;
    width: min(12rem, 100%);
    height: auto;
    margin-left: auto;
}
div[class*="st-key-theme-toggle-container"] button {
    width: 100%;
    min-height: 2.55rem;
    height: auto;
    padding: 0.45rem 0.85rem;
    border: 1.5px solid #7b69af !important;
    border-radius: 999px;
    background: linear-gradient(90deg, #eee9ff, #d7f2ef) !important;
    box-shadow: inset 0 1px 2px rgba(255,255,255,0.85), 0 5px 12px rgba(91, 75, 142, 0.16) !important;
    color: #423a60 !important;
    font-family: 'Poppins', Georgia, serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    line-height: 1.1;
    text-align: center;
    text-transform: uppercase;
    white-space: nowrap;
    opacity: 1;
}
div[class*="st-key-theme-toggle-container"] button:hover {
    transform: translateY(-1px);
    background: linear-gradient(90deg, #e4ddff, #c7ece8) !important;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: #423a60;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin: 1.8rem 0 0.7rem 0;
}
.section-title::before {
    content: "";
    width: 0.32rem;
    height: 1.25rem;
    flex: 0 0 auto;
    border-radius: 999px;
    background: linear-gradient(180deg, #ff9bab, #a995ec 52%, #72d4cc);
    box-shadow: 0 2px 8px rgba(144, 120, 207, 0.30);
}
.section-title::after {
    content: "";
    height: 1px;
    flex: 1;
    background: linear-gradient(90deg, rgba(169, 149, 236, 0.65), rgba(114, 212, 204, 0));
}

/* Gold bordered content cards — darker, clearly visible border in light mode */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.78), rgba(255, 252, 254, 0.58));
    border: 2px solid #241b42 !important;
    border-radius: 22px;
    box-shadow: 0 12px 24px rgba(125, 110, 165, 0.13), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}
div[class*="st-key-current-parameters"],
div[class*="st-key-prediction-result"],
div[class*="st-key-model-explainability"],
div[class*="st-key-current-profile"],
div[class*="st-key-prediction-history"] {
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
/* Light-mode hover: purple glow */
div[class*="st-key-current-parameters"]:hover,
div[class*="st-key-prediction-result"]:hover,
div[class*="st-key-model-explainability"]:hover,
div[class*="st-key-current-profile"]:hover,
div[class*="st-key-prediction-history"]:hover {
    border-color: #6f5aa8 !important;
    transform: translateY(-6px) perspective(950px) rotateX(2deg) rotateY(-0.8deg);
    box-shadow: 0 28px 42px rgba(124, 108, 173, 0.28), 0 10px 22px rgba(255, 174, 194, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.95);
}
.parameter-card-title {
    font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif;
    color: #665987;
    font-size: 0.83rem;
    font-weight: 800;
    letter-spacing: 0.65px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

/* Premium info box */
div[data-testid="stAlertContainer"] {
    background: linear-gradient(180deg, rgba(36,50,56,0.72), rgba(23,31,36,0.78)) !important;
    backdrop-filter: blur(12px);
    border-radius: 20px !important;
    border: 1px solid rgba(210, 181, 118, 0.26) !important;
    color: #ecf0ee !important;
    margin-top: 0.75rem !important;
    box-shadow: 0 14px 28px rgba(0,0,0,0.18);
}

/* Dataframe / table container */
div[data-testid="stDataFrame"] {
    border-radius: 30px;
    overflow: hidden;
    box-shadow: 0 20px 30px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(154, 192, 190, 0.2);
}

/* Light-theme component overrides */
label, .stCaption, [data-testid="stMarkdownContainer"] p {
    color: #5e5878 !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #b9a9f4 !important;
    border-color: #ffffff !important;
    box-shadow: 0 3px 8px rgba(123, 105, 175, 0.26);
}
div[data-testid="stTickBar"] {
    background: linear-gradient(90deg, #ffbdc8, #c1b2ff, #9ee5df) !important;
}
div[data-testid="stAlertContainer"] {
    background: rgba(255, 255, 255, 0.66) !important;
    border-color: rgba(193, 178, 255, 0.42) !important;
    color: #4e4769 !important;
    box-shadow: 0 12px 24px rgba(125, 110, 165, 0.12);
}
div[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.62);
    box-shadow: 0 12px 24px rgba(125, 110, 165, 0.12);
    border-color: rgba(193, 178, 255, 0.35);
}
div[data-testid="stDataFrame"] [role="grid"],
div[data-testid="stDataFrame"] [role="row"],
div[data-testid="stDataFrame"] [role="gridcell"] {
    background: rgba(255, 255, 255, 0.48) !important;
    color: #4e4769 !important;
}
div[data-testid="stDataFrame"] [role="columnheader"] {
    background: linear-gradient(90deg, rgba(255, 193, 204, 0.58), rgba(193, 178, 255, 0.55), rgba(158, 229, 223, 0.52)) !important;
    color: #403758 !important;
    font-weight: 800 !important;
}
div[data-testid="stDataFrame"] [role="gridcell"]:hover {
    background: rgba(224, 218, 255, 0.42) !important;
}

/* Dark mode — warm black glass with the ember glow of the reference image */
.stApp.dark-mode {
    background:
        radial-gradient(ellipse 46% 44% at 15% 48%, rgba(164, 72, 18, 0.68), transparent 72%),
        radial-gradient(ellipse 36% 32% at 83% 67%, rgba(244, 94, 18, 0.50), transparent 68%),
        radial-gradient(ellipse 58% 36% at 53% 55%, rgba(77, 25, 12, 0.42), transparent 74%),
        linear-gradient(135deg, #050505 0%, #0d0806 48%, #180a04 100%);
    color: #f3e9df;
}
.stApp.dark-mode::before {
    background: radial-gradient(
        circle 16rem at var(--cursor-x, 50%) var(--cursor-y, 45%),
        rgba(255, 166, 93, 0.22) 0%,
        rgba(213, 81, 23, 0.12) 34%,
        rgba(0, 0, 0, 0) 72%
    );
}
/* Dark-mode cards: darker warm-amber border, clearly visible against the black glass */
.stApp.dark-mode .hero,
.stApp.dark-mode div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(23, 20, 19, 0.88), rgba(7, 7, 7, 0.82));
    border: 1.5px solid #f0954f !important;
    box-shadow: 0 18px 34px rgba(0, 0, 0, 0.48), inset 0 1px 0 rgba(255, 208, 166, 0.08);
}
/* Dark-mode hover: orange/ember glow instead of purple */
.stApp.dark-mode div[class*="st-key-current-parameters"]:hover,
.stApp.dark-mode div[class*="st-key-prediction-result"]:hover,
.stApp.dark-mode div[class*="st-key-model-explainability"]:hover,
.stApp.dark-mode div[class*="st-key-current-profile"]:hover,
.stApp.dark-mode div[class*="st-key-prediction-history"]:hover {
    border-color: #ffb066 !important;
    box-shadow: 0 28px 42px rgba(196, 92, 24, 0.34), 0 10px 22px rgba(240, 138, 60, 0.30), inset 0 1px 0 rgba(255, 234, 212, 0.14);
}
.stApp.dark-mode .hero h1,
.stApp.dark-mode .risk-card p,
.stApp.dark-mode .section-title,
.stApp.dark-mode .parameter-card-title,
.stApp.dark-mode label,
.stApp.dark-mode .stCaption,
.stApp.dark-mode [data-testid="stMarkdownContainer"] p {
    color: #f4e8dc !important;
}
.stApp.dark-mode .hero p {
    color: #cbb8aa;
}
.stApp.dark-mode .section-title::before {
    background: linear-gradient(180deg, #ffc08d, #e97a2f 55%, #a64814);
    box-shadow: 0 2px 8px rgba(233, 122, 47, 0.34);
}
.stApp.dark-mode .section-title::after {
    background: linear-gradient(90deg, rgba(240, 149, 79, 0.72), rgba(240, 149, 79, 0));
}
.stApp.dark-mode .stButton > button {
    background: linear-gradient(135deg, #a64814, #e97a2f 58%, #f1ad76);
    color: #1a0c05;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255, 234, 212, 0.30);
}
.stApp.dark-mode .stButton > button:hover {
    background: linear-gradient(135deg, #c35a19, #f18a3d 58%, #ffc08d);
}
/* The invisible theme-toggle button must stay transparent even under the dark-mode button skin above */
.stApp.dark-mode div[class*="st-key-theme-toggle-container"] button,
.stApp.dark-mode div[class*="st-key-theme-toggle-container"] button:hover {
    background: linear-gradient(90deg, #57220d, #bd5724) !important;
    border-color: #f6a45d !important;
    box-shadow: inset 0 1px 2px rgba(255, 225, 197, 0.25), 0 5px 12px rgba(0, 0, 0, 0.34) !important;
    color: #fff1e5 !important;
}
.stApp.dark-mode div[data-testid="stAlertContainer"],
.stApp.dark-mode div[data-testid="stDataFrame"],
.stApp.dark-mode div[data-testid="stDataFrame"] [role="grid"],
.stApp.dark-mode div[data-testid="stDataFrame"] [role="row"],
.stApp.dark-mode div[data-testid="stDataFrame"] [role="gridcell"] {
    background: rgba(16, 13, 12, 0.82) !important;
    border-color: rgba(255, 179, 117, 0.24) !important;
    color: #f0dfd3 !important;
}
.stApp.dark-mode div[data-testid="stDataFrame"] [role="columnheader"] {
    background: linear-gradient(90deg, rgba(123, 47, 14, 0.85), rgba(204, 88, 25, 0.72)) !important;
    color: #fff1e6 !important;
}
.stApp.dark-mode .risk-card {
    border-color: rgba(255, 190, 139, 0.26);
    box-shadow: 0 16px 30px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255,255,255,0.06);
}
.stApp.dark-mode .risk-low { background: linear-gradient(135deg, rgba(23, 90, 66, 0.64), rgba(8, 24, 18, 0.78)); }
.stApp.dark-mode .risk-medium { background: linear-gradient(135deg, rgba(131, 68, 15, 0.65), rgba(42, 20, 6, 0.78)); }
.stApp.dark-mode .risk-high { background: linear-gradient(135deg, rgba(128, 35, 21, 0.70), rgba(43, 12, 8, 0.80)); }

footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# THEME TOGGLE — native Streamlit control with elastic styling
# ============================================================
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


_, theme_card_col = st.columns([12, 2.6])
with theme_card_col:
    theme_label = "Light mode" if is_dark_mode else "Dark mode"
    theme_icon = "☀" if is_dark_mode else "☾"
    st.button(
        f"{theme_icon}  Switch to {theme_label}",
        key="theme-toggle-container",
        on_click=toggle_theme,
        help=f"Switch to {theme_label}.",
        use_container_width=True,
    )

# ============================================================
# THEME CLASS + CURSOR GLOW (injected once, hardened)
# ============================================================
components.html(
    """
    <script>
    (function() {
        const darkMode = DARK_MODE;
        try {
            const doc = window.parent.document;
            const app = doc.querySelector(".stApp");
            if (app) { app.classList.toggle("dark-mode", darkMode); }

            const root = doc.documentElement;
            if (root.__cursorGlowHandler) {
                doc.removeEventListener("mousemove", root.__cursorGlowHandler);
            }
            root.__cursorGlowHandler = function(event) {
                root.style.setProperty("--cursor-x", event.clientX + "px");
                root.style.setProperty("--cursor-y", event.clientY + "px");
            };
            doc.addEventListener("mousemove", root.__cursorGlowHandler, { passive: true });
        } catch (err) {
            // Fails silently in sandboxed / restricted embeds — cosmetic only.
        }
    })();
    </script>
    """.replace("DARK_MODE", str(is_dark_mode).lower()),
    height=0,
)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_model_and_scaler()

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Landslide Risk Prediction Dashboard</h1>
    <p>Estimate landslide probability from rainfall, slope, soil saturation and vegetation cover.</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.warning(
        f"⚠️ Model/scaler could not be loaded from **{MODEL_PATH}** / **{SCALER_PATH}**.  "
        f"The dashboard UI will still work, but predictions are disabled until valid "
        f"`joblib` files are available.\n\nDetails: `{load_error}`"
    )

# ============================================================
# CURRENT PARAMETER OVERVIEW
# ============================================================
with st.container(border=True, key="current-parameters"):
    st.markdown('<div class="section-title">Current Parameters</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders to match the site conditions you want to evaluate.")
    values = {}
    input_cols = st.columns(2)
    for index, col_name in enumerate(INPUT_COLUMNS):
        lo, hi, default, step, unit, fmt = PARAM_RANGES[col_name]
        label = col_name.replace("_", " ")
        display_label = f"{label} ({unit})" if unit else label
        with input_cols[index % 2]:
            with st.container(border=True, key=f"input-{col_name.lower()}"):
                st.markdown(f'<div class="parameter-card-title">{display_label}</div>', unsafe_allow_html=True)
                values[col_name] = st.slider(
                    display_label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    format=fmt,
                    label_visibility="collapsed",
                )

    action_col1, action_col2, _ = st.columns([1, 1, 2])
    with action_col1:
        predict_clicked = st.button("🔮 Predict Landslide Risk", use_container_width=True)
    with action_col2:
        reset_clicked = st.button("↺ Reset History", use_container_width=True)

# ============================================================
# SESSION STATE FOR PREDICTION HISTORY
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "has_prediction" not in st.session_state:
    st.session_state.has_prediction = False

if reset_clicked:
    st.session_state.history = []
    st.session_state.has_prediction = False

# ============================================================
# PREDICTION
# ============================================================
def get_risk_bucket(prob: float):
    if prob < 0.34:
        return "Low", "risk-low", "#4ade80"
    elif prob < 0.67:
        return "Medium", "risk-medium", "#fbbf24"
    else:
        return "High", "risk-high", "#f87171"


FEATURE_LABELS = {
    "Rainfall_mm": "rainfall",
    "Slope_Angle": "slope angle",
    "Soil_Saturation": "soil saturation",
    "Vegetation_Cover": "vegetation cover",
}


def relative_range_level(value: float, lower: float, upper: float) -> str:
    """Classify an input by thirds of its synthetic training-data range."""
    position = (value - lower) / (upper - lower)
    if position < 1 / 3:
        return "Low"
    if position < 2 / 3:
        return "Medium"
    return "High"


def coefficient_explanations(model):
    """Return directional, not magnitude-based, Logistic Regression explanations."""
    if not hasattr(model, "coef_"):
        return []

    coefficients = np.asarray(model.coef_).reshape(-1)
    if len(coefficients) != len(INPUT_COLUMNS):
        return []

    explanations = []
    for feature, coefficient in zip(INPUT_COLUMNS, coefficients):
        label = FEATURE_LABELS[feature]
        if coefficient > 0:
            explanations.append(
                f"Higher {label} is associated with increased predicted landslide risk."
            )
        elif coefficient < 0:
            explanations.append(
                f"Higher {label} is associated with lower predicted landslide risk in this model."
            )
        else:
            explanations.append(
                f"{label.capitalize()} has no directional association in this model."
            )
    return explanations


if predict_clicked:
    if model is None or scaler is None:
        st.error("Cannot predict — model or scaler failed to load. Check the file paths at the top of the script.")
    else:
        input_df = pd.DataFrame([[values[c] for c in INPUT_COLUMNS]], columns=INPUT_COLUMNS)
        try:
            scaled_input = scaler.transform(input_df)
            pred_class = model.predict(scaled_input)[0]

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(scaled_input)[0][-1])
            else:
                prob = float(pred_class)

            risk_label, risk_class, risk_color = get_risk_bucket(prob)

            st.session_state.history.append({
                **values,
                "Risk_Probability": round(prob * 100, 1),
                "Risk_Level": risk_label,
            })
            st.session_state.has_prediction = True

            # ---------- RESULTS LAYOUT ----------
            with st.container(border=True, key="prediction-result"):
                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1.3])

                with res_col1:
                    st.markdown(f"""
                    <div class="risk-card {risk_class}">
                        <h2>{risk_label} Risk</h2>
                        <p>Estimated Landslide Probability: <b>{prob*100:.1f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                    advice = {
                        "Low": "Conditions appear stable. Routine monitoring is sufficient.",
                        "Medium": "Elevated risk. Increase monitoring frequency and watch for weather changes.",
                        "High": "Significant risk. Consider issuing an alert and restricting access to the slope area.",
                    }[risk_label]
                    st.info(f"💡 **Recommendation:** {advice}")

                with res_col2:
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%", "font": {"size": 40, "color": plot_text_color}},
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": plot_text_color},
                            "bar": {"color": risk_color},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 1,
                            "bordercolor": "rgba(255,255,255,0.2)",
                            "steps": [
                                {"range": [0, 34], "color": "rgba(74,222,128,0.25)"},
                                {"range": [34, 67], "color": "rgba(251,191,36,0.25)"},
                                {"range": [67, 100], "color": "rgba(248,113,113,0.25)"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.8,
                                "value": prob * 100,
                            },
                        },
                    ))
                    gauge_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": plot_text_color},
                        height=280,
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)

            # ---------- MODEL EXPLAINABILITY ----------
            with st.container(border=True, key="model-explainability"):
                st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
                st.caption(
                    "These directions come from the Logistic Regression coefficients after feature "
                    "standardization. They describe this model's associations, not causal scientific evidence."
                )

                explanations = coefficient_explanations(model)
                if explanations:
                    for explanation in explanations:
                        st.write(f"• {explanation}")
                else:
                    st.info("Coefficient-based explanations are unavailable for the loaded model.")

                profile_rows = []
                for feature in INPUT_COLUMNS:
                    low, high, _, _, unit, display_format = PARAM_RANGES[feature]
                    current_value = display_format % values[feature]
                    training_range = f"{display_format % low}–{display_format % high}{unit}"
                    profile_rows.append({
                        "Input": feature.replace("_", " "),
                        "Current input": f"{current_value}{unit}",
                        "Synthetic training-data range": training_range,
                        "Relative level": relative_range_level(values[feature], low, high),
                    })

                st.caption("Current inputs positioned within the synthetic training-data ranges.")
                st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# CURRENT PARAMETER PROFILE AND PREDICTION HISTORY
# ============================================================
if st.session_state.has_prediction and st.session_state.history:
    profile_col, history_col = st.columns(2)

    with profile_col:
        with st.container(border=True, key="current-profile"):
            st.markdown('<div class="section-title">Current Parameter Profile</div>', unsafe_allow_html=True)
            norm_vals = []
            for c in INPUT_COLUMNS:
                lo, hi, *_ = PARAM_RANGES[c]
                norm_vals.append((values[c] - lo) / (hi - lo) if hi != lo else 0)

            radar_labels = [c.replace("_", " ") for c in INPUT_COLUMNS]
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor=plot_fill_color,
                line=dict(color=plot_accent_color, width=1.5),
                name="Current input",
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor=plot_background,
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showticklabels=False,
                        showline=False,
                        ticks="",
                        ticklen=0,
                        gridcolor="rgba(120, 100, 165, 0.14)",
                        linecolor="rgba(120, 100, 165, 0.14)",
                    ),
                    angularaxis=dict(
                        showline=False,
                        ticks="",
                        ticklen=0,
                        gridcolor="rgba(120, 100, 165, 0.14)",
                        linecolor="rgba(120, 100, 165, 0.14)",
                        color=plot_label_color,
                    ),
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": plot_label_color, "family": "Poppins, Georgia, serif"},
                height=300,
                margin=dict(l=30, r=30, t=20, b=20),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    with history_col:
        with st.container(border=True, key="prediction-history"):
            st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
            hist_df = pd.DataFrame(st.session_state.history)

            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=list(range(1, len(hist_df) + 1)),
                y=hist_df["Risk_Probability"],
                mode="lines+markers",
                line=dict(color=plot_accent_color, width=3),
                marker=dict(size=11, color=plot_marker_color, line=dict(color="#ffffff", width=2)),
                fill="tozeroy",
                fillcolor=plot_fill_color,
                name="Estimated Landslide Probability",
                hovertemplate="Prediction %{x}<br>Estimated probability: %{y:.1f}%<extra></extra>",
            ))
            trend_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=plot_background,
                font={"color": plot_label_color, "family": "Poppins, Georgia, serif"},
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                yaxis=dict(
                    title="Estimated probability (%)",
                    title_font=dict(color=plot_text_color, size=14),
                    gridcolor=plot_grid_color,
                    zerolinecolor=plot_grid_color,
                    tickfont=dict(color=plot_label_color),
                    range=[0, 100],
                ),
                xaxis=dict(
                    title="Prediction #",
                    title_font=dict(color=plot_text_color, size=14),
                    gridcolor=plot_grid_color,
                    zeroline=False,
                    tickfont=dict(color=plot_label_color),
                    dtick=1,
                ),
            )
            st.plotly_chart(trend_fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#8fae94; font-size:0.8rem; margin-top:2rem;">
    Prototype model trained on synthetic data. Results are for demonstration and research purposes and should not be used as a substitute for professional hazard assessment.<br>
    Built with Streamlit · Model served via joblib
</div>
""", unsafe_allow_html=True)
