import streamlit as st
import joblib 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "landslide_model4.pkl"
SCALER_PATH = "landslide_scaler4.pkl"

INPUT_COLUMNS = ["Rainfall_mm", "Slope_Angle", "Soil_Saturation", "Vegetation_Cover"]

# (min, max, default, step, unit, display_format) — used for sliders + radar normalization
PARAM_RANGES = {
    "Rainfall_mm":      (0.0,  300.0, 100.0, 5.0,  "mm", "%.0f"),
    "Slope_Angle":      (5.0,  80.0,  30.0,  1.0,  "°",  "%.0f"),
    "Soil_Saturation":  (0.10, 1.00,  0.50,  0.01, "",   "%.2f"),
    "Vegetation_Cover": (0.00, 1.00,  0.50,  0.01, "",   "%.2f"),
}

st.set_page_config(
    page_title="Landslide Risk Predictor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING — premium dark mountain dashboard theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Cormorant+Garamond:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Premium dark background with layered mountain gradients */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(148, 119, 78, 0.18), transparent 26%),
        radial-gradient(circle at bottom right, rgba(19, 94, 90, 0.20), transparent 30%),
        linear-gradient(135deg, #090d11 0%, #11191d 18%, #162028 42%, #1a2628 100%);
    color: #edf3f2;
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
    background: linear-gradient(135deg, rgba(22, 31, 35, 0.82), rgba(40, 49, 55, 0.72));
    border: 1px solid rgba(196, 174, 126, 0.25);
    box-shadow: 0 30px 50px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.06);
    margin-bottom: 1.6rem;
}
.hero h1 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-weight: 700;
    font-size: 3rem;
    letter-spacing: 0.01em;
    margin: 0;
    line-height: 0.96;
    color: #f4efe7;
}
.hero p {
    color: #c0c9c6;
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
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 0.5rem;
    max-width: 400px;
    max-height: 400px;
    width: 100%;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 18px 30px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255,255,255,0.08);
}
.risk-card h2 {
    margin: 0;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.9rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1;
}
.risk-card p {
    color: #ecf0ee;
    margin-top: 0.45rem;
    font-size: 0.95rem;
}
.risk-low    { background: linear-gradient(135deg, rgba(34,120,86,0.20), rgba(17,35,30,0.18)); border-color: rgba(97, 202, 144, 0.55); }
.risk-low h2 { color: #7ae5a7; }
.risk-medium { background: linear-gradient(135deg, rgba(170,118,42,0.20), rgba(34,27,15,0.18)); border-color: rgba(235,179,80,0.55); }
.risk-medium h2 { color: #f3c76e; }
.risk-high   { background: linear-gradient(135deg, rgba(148,49,49,0.22), rgba(33,16,18,0.18)); border-color: rgba(247,111,111,0.60); }
.risk-high h2 { color: #ff8d8d; }

/* Premium action buttons */
.stButton>button {
    background: linear-gradient(135deg, #d4b579, #b68f4d);
    color: #121a1d;
    border: none;
    border-radius: 100px;
    padding: 0.7rem 1.25rem;
    font-weight: 800;
    letter-spacing: 0.2px;
    box-shadow: 0 12px 24px rgba(22, 24, 27, 0.35), inset 0 1px 0 rgba(255,255,255,0.35);
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #e4c98c, #c89d5c);
    transform: translateY(-2px);
    box-shadow: 0 16px 28px rgba(183, 146, 84, 0.24);
    color: #11181c;
}

/* Section headers — warm metallic accent bar */
.section-title {
    font-size: 0.9rem;
    font-weight: 800;
    color: #f2efe8;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    margin: 1.8rem 0 0.7rem 0;
    border-left: 4px solid #d4b579;
    padding-left: 0.7rem;
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

footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


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
# SIDEBAR — INPUTS
# ============================================================
with st.sidebar:
    st.markdown("### 🧭 Input Parameters")
    st.caption("Adjust the sliders to match the site conditions you want to evaluate.")

    values = {}
    for col in INPUT_COLUMNS:
        lo, hi, default, step, unit, fmt = PARAM_RANGES[col]
        label = col.replace("_", " ")
        label = f"{label} ({unit})" if unit else label
        values[col] = st.slider(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=float(step),
            format=fmt,
        )

    st.markdown("---")
    predict_clicked = st.button("🔮 Predict Landslide Risk", use_container_width=True)
    reset_clicked = st.button("↺ Reset History", use_container_width=True)

# ============================================================
# CURRENT PARAMETER OVERVIEW
# ============================================================
st.markdown('<div class="section-title">Current Parameters</div>', unsafe_allow_html=True)
cols = st.columns(4)
icons = {"Rainfall_mm": "🌧️", "Slope_Angle": "⛰️", "Soil_Saturation": "💧", "Vegetation_Cover": "🌿"}
for c, col_name in zip(cols, INPUT_COLUMNS):
    lo, hi, default, step, unit, fmt = PARAM_RANGES[col_name]
    with c:
        st.markdown(f"""
        <div class="param-card">
            <div class="label">{icons[col_name]} {col_name.replace('_',' ')}</div>
            <div class="value">{fmt % values[col_name]}{unit}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SESSION STATE FOR PREDICTION HISTORY
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

if reset_clicked:
    st.session_state.history = []

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

            # ---------- RESULTS LAYOUT ----------
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 1.3])

            with res_col1:
                st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <h2>{risk_label} Risk</h2>
                    <p>Predicted probability of landslide: <b>{prob*100:.1f}%</b></p>
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
                    number={"suffix": "%", "font": {"size": 40, "color": "#eafbf0"}},
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#eafbf0"},
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
                    font={"color": "#eafbf0"},
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=10),
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            # ---------- RADAR CHART OF NORMALIZED INPUTS ----------
            st.markdown('<div class="section-title">Parameter Profile</div>', unsafe_allow_html=True)
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
                fillcolor="rgba(183,242,60,0.35)",
                line=dict(color="#b7f23c", width=2),
                name="Current input",
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.15)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)", color="#eaffd0"),
                ),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#eaffd0", "family": "Inter"},
                height=350,
                margin=dict(l=40, r=40, t=30, b=30),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# PREDICTION HISTORY
# ============================================================
if st.session_state.history:
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        y=hist_df["Risk_Probability"],
        mode="lines+markers",
        line=dict(color="#b7f23c", width=2),
        marker=dict(size=8, color="#eaffd0"),
        name="Risk Probability (%)",
    ))
    trend_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eaffd0", "family": "Inter"},
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title="Risk %", gridcolor="rgba(255,255,255,0.1)", range=[0, 100]),
        xaxis=dict(title="Prediction #", gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(trend_fig, use_container_width=True)
else:
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    st.caption("No predictions yet — adjust the sliders in the sidebar and click **Predict Landslide Risk**.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; color:#8fae94; font-size:0.8rem; margin-top:2rem;">
    Built with Streamlit · Model served via joblib
</div>
""", unsafe_allow_html=True)