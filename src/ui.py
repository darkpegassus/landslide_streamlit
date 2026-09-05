"""Theme-aware visual components for the landslide dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

FEATURE_NAMES = {"numerical__Slope_Angle": "Slope angle", "numerical__Vegetation_Cover": "Vegetation cover", "numerical__Soil_Erosion_Rate": "Soil erosion rate", "categorical__Geology_Type_Colluvium": "Geology type: Colluvium", "numerical__Seismic_PGA_g": "Seismic PGA", "numerical__Pore_Pressure_Ratio": "Pore pressure ratio", "numerical__Soil_Saturation": "Soil saturation", "numerical__Rainfall_3Day": "Rainfall (3 day)"}


def theme(dark: bool) -> dict[str, str]:
    return ({"text": "#f4e8dc", "muted": "#cbb8aa", "accent": "#f0954f", "grid": "rgba(240,149,79,.24)", "plot": "rgba(16,10,8,.48)", "bar": "#e98945", "marker": "#ffc08d"} if dark else {"text": "#3f3656", "muted": "#655d7c", "accent": "#9a86dd", "grid": "rgba(113,94,165,.18)", "plot": "rgba(235,229,250,.82)", "bar": "#8d7ddd", "marker": "#f18da2"})


def inject_styles(dark: bool) -> None:
    """Reference-style palettes, a self-contained pill switch, and hover cards."""
    c = theme(dark)
    tokens = {
        "BACKGROUND": (
            "radial-gradient(ellipse 46% 44% at 15% 48%, rgba(164, 72, 18, 0.68), transparent 72%), "
            "radial-gradient(ellipse 36% 32% at 83% 67%, rgba(244, 94, 18, 0.50), transparent 68%), "
            "linear-gradient(135deg, #050505 0%, #0d0806 48%, #180a04 100%)"
        ) if dark else (
            "radial-gradient(ellipse 58% 29% at 12% 48%, rgba(255, 181, 194, 0.78), transparent 72%), "
            "radial-gradient(ellipse 52% 34% at 50% 50%, rgba(180, 170, 250, 0.70), transparent 74%), "
            "radial-gradient(ellipse 50% 38% at 83% 52%, rgba(163, 235, 231, 0.72), transparent 76%), "
            "radial-gradient(ellipse 70% 42% at 48% 58%, rgba(255, 244, 234, 0.84), transparent 74%), "
            "linear-gradient(180deg, #fffdfb 0%, #faf8ff 44%, #f4fcfc 100%)"
        ),
        "CARD": "linear-gradient(145deg, rgba(23, 20, 19, 0.88), rgba(7, 7, 7, 0.82))" if dark else "linear-gradient(145deg, rgba(255, 255, 255, 0.78), rgba(255, 252, 254, 0.58))",
        "CARD_BORDER": "#f0954f" if dark else "#241b42",
        "TEXT": c["text"], 
        "MUTED": c["muted"], 
        "ACCENT": c["accent"], 
        "GRID": c["grid"], 
        "METRIC": "rgba(85,39,21,.48)" if dark else "rgba(238,232,255,.72)",
        "GLOW": "rgba(255,166,93,.22)" if dark else "rgba(255,255,255,.80)", 
        "HOVER": "#ffb066" if dark else "#6f5aa8", 
        "SHADOW": "0 28px 42px rgba(196,92,24,.34), 0 10px 22px rgba(240,138,60,.30)" if dark else "0 28px 42px rgba(124,108,173,.28), 0 10px 22px rgba(255,174,194,.22)",
        "TRACK": "linear-gradient(90deg, #57220d, #bd5724)" if dark else "linear-gradient(90deg, #eee9ff, #d7f2ef)",
        "TRACK_HOVER": "linear-gradient(90deg, #57220d, #bd5724)" if dark else "linear-gradient(90deg, #e4ddff, #c7ece8)", 
        "SWITCH_BORDER": "#f6a45d" if dark else "#7b69af", 
        "SWITCH_TEXT": "#fff1e5" if dark else "#423a60",
        "SWITCH_SHADOW": "inset 0 1px 2px rgba(255, 225, 197, 0.25), 0 5px 12px rgba(0, 0, 0, 0.34)" if dark else "inset 0 1px 2px rgba(255,255,255,0.85), 0 5px 12px rgba(91, 75, 142, 0.16)",
        "LOW_A": "#275a45" if dark else "#bdebd6", 
        "LOW_B": "#10251c" if dark else "#e8f8ee", 
        "LOW_T": "#d9f7e7" if dark else "#164b38", 
        "MED_A": "#765016" if dark else "#ffd477", 
        "MED_B": "#2e1c08" if dark else "#fff1cc", 
        "MED_T": "#ffe9b2" if dark else "#6b3d04", 
        "HIGH_A": "#722f31" if dark else "#ffb5ba", 
        "HIGH_B": "#2d1014" if dark else "#ffe1e2", 
        "HIGH_T": "#ffe1e2" if dark else "#6d1722",
        "BTN_BG": "linear-gradient(100deg, #b95721, #f1ad76)" if dark else "linear-gradient(135deg, #ffb9c4, #c1b2ff 54%, #9de4df)",
        "BTN_HOVER": "linear-gradient(100deg, #c35a19, #f18a3d)" if dark else "linear-gradient(135deg, #ffc7cf, #d1c6ff 54%, #b6f1eb)",
        "BTN_TEXT": "#1b0c05" if dark else "#403758",
        "BTN_SHADOW": "0 8px 18px rgba(0, 0, 0, 0.24)" if dark else "0 12px 22px rgba(133, 108, 172, 0.24), inset 0 1px 0 rgba(255,255,255,0.72)",
    }
    
    css = '''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Poppins:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: Poppins, Arial, sans-serif;
        }
        
        .stApp {
            background: __BACKGROUND__;
            color: __TEXT__;
            position: relative;
            isolation: isolate;
        }
        
        .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background: radial-gradient(circle 16rem at var(--cursor-x, 50%) var(--cursor-y, 40%), __GLOW__ 0%, transparent 72%);
            transition: background 0.12s ease-out;
        }
        
        .stApp > div {
            position: relative;
            z-index: 1;
        }
        
        .block-container {
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 1.4rem;
        }
        
        header[data-testid="stHeader"] {
            background: transparent;
        }
        
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: __CARD__;
            border: 2px solid __CARD_BORDER__ !important;
            border-radius: 18px;
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.22);
        }
        
        div[class*="st-key-current-parameters"],
        div[class*="st-key-prediction-result"],
        div[class*="st-key-feature-importance"],
        div[class*="st-key-system-information"],
        div[class*="st-key-current-profile"],
        div[class*="st-key-prediction-history"] {
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            transform: perspective(950px) translateZ(0);
        }
        
        div[class*="st-key-current-parameters"]:hover,
        div[class*="st-key-prediction-result"]:hover,
        div[class*="st-key-feature-importance"]:hover,
        div[class*="st-key-system-information"]:hover,
        div[class*="st-key-current-profile"]:hover,
        div[class*="st-key-prediction-history"]:hover {
            border-color: __HOVER__ !important;
            transform: translateY(-6px) perspective(950px) rotateX(2deg) rotateY(-0.8deg);
            box-shadow: __SHADOW__;
        }
        
        .hero {
            padding: 1.6rem 2rem;
            margin: -1.8rem 0 1rem; /* The negative top margin closes the gap */
            border: 2px solid __ACCENT__;
            border-radius: 32px;
            background: __CARD__;
            box-shadow: 0 16px 34px rgba(0,0,0,.26);
        }
        
        .hero h1 {
            margin: 0;
            color: __TEXT__;
            font: 700 3.05rem/1 'Barlow Condensed', sans-serif;
        }
        
        .hero p {
            margin: 0.55rem 0 0;
            color: __MUTED__;
            font-size: 0.94rem;
        }
        
        .section-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 0.8rem;
            color: __TEXT__;
            font: 800 1.05rem 'Barlow Condensed', sans-serif;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .section-title:before {
            content: "";
            width: 5px;
            height: 1.35rem;
            border-radius: 5px;
            background: linear-gradient(#ffc08d, __ACCENT__);
            box-shadow: 0 0 9px __ACCENT__;
        }
        
        .section-title:after {
            content: "";
            height: 1px;
            flex: 1;
            background: linear-gradient(90deg, __ACCENT__, transparent);
        }
        
        .input-group-title {
            margin: 0.15rem 0 0.25rem;
            padding-bottom: 0.35rem;
            color: __ACCENT__;
            border-bottom: 1px solid __GRID__;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }
        
        label {
            color: __TEXT__ !important;
            font-size: 0.79rem !important;
            font-weight: 600 !important;
        }
        
        /* Dynamic Primary Button Styling */
        .stButton > button {
            min-height: 2.6rem;
            border: 0;
            border-radius: 999px;
            background: __BTN_BG__;
            box-shadow: __BTN_SHADOW__;
            color: __BTN_TEXT__;
            font-weight: 800;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            background: __BTN_HOVER__;
            filter: brightness(1.08);
        }
        
        div[class*="st-key-reset"] button {
            background: transparent;
            border: 1px solid __ACCENT__;
            color: __TEXT__;
            box-shadow: none;
        }
        
        /* Theme switch: wrapping text and compact sizing */
        div[class*="st-key-theme-toggle-container"] {
            position: relative;
            z-index: 999999;
            width: 100%;
            max-width: 8.5rem; 
            margin-left: auto;
        }
        
        div[class*="st-key-theme-toggle-container"] button {
            width: 100%;
            font-size: 0.01rem;
            min-height: 2.4rem;
            padding: 0.15rem 0.6rem;
            border: 1.5px solid __SWITCH_BORDER__ ;
            border-radius: 999px;
            background: __TRACK__ ;
            color: __SWITCH_TEXT__ ;
            font-family: Poppins, Georgia, serif;
            font-weight: 700 ;
            letter-spacing: 0.01em;
            line-height: 1.25 ; 
            text-align: center;
            text-transform: uppercase;
            box-shadow: __SWITCH_SHADOW__ ;
            white-space: normal;
        }
        
        div[class*="st-key-theme-toggle-container"] button:hover {
            transform: translateY(-1px);
            background: __TRACK_HOVER__ !important;
            filter: brightness(1.08);
        }
        
        .risk-summary {
            display: flex;
            min-height: 178px;
            flex-direction: column;
            justify-content: center;
            padding: 1.1rem 1.25rem;
            border-radius: 18px;
        }
        
        .risk-low {
            background: linear-gradient(135deg, __LOW_A__, __LOW_B__);
            color: __LOW_T__;
        }
        
        .risk-medium {
            background: linear-gradient(135deg, __MED_A__, __MED_B__);
            color: __MED_T__;
        }
        
        .risk-high {
            background: linear-gradient(135deg, __HIGH_A__, __HIGH_B__);
            color: __HIGH_T__;
        }
        
        .risk-summary .risk-name {
            font: 800 2rem/1 'Barlow Condensed', sans-serif;
        }
        
        .risk-summary .risk-number {
            margin: 0.35rem 0 0.05rem;
            font-size: 2.3rem;
            font-weight: 800;
        }
        
        .risk-summary .risk-copy {
            font-size: 0.83rem;
            font-weight: 600;
        }
        
        .decision-pill {
            display: inline-block;
            margin-top: 0.8rem;
            padding: 0.3rem 0.55rem;
            border-radius: 999px;
            background: rgba(0, 0, 0, 0.12);
            font-size: 0.74rem;
            font-weight: 700;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }
        
        .metric {
            padding: 0.65rem 0.75rem;
            border-radius: 11px;
            background: __METRIC__;
        }
        
        .metric span {
            display: block;
            color: __MUTED__;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        
        .metric b {
            display: block;
            margin-top: 0.15rem;
            color: __TEXT__;
            font-size: 0.88rem;
        }
        
        div[data-testid="stDataFrame"] {
            overflow: hidden;
            border: 1px solid __GRID__;
            border-radius: 10px;
        }
        
        .footer {
            margin-top: 1rem;
            color: __MUTED__;
            font-size: 0.75rem;
            text-align: center;
        }
    </style>
    '''
    
    for key, value in tokens.items(): 
        css = css.replace(f"__{key}__", value)
        
    st.markdown(css, unsafe_allow_html=True)


def inject_cursor_glow() -> None:
    components.html('''<script>(()=>{const d=window.parent.document,r=d.documentElement;if(r.__landslideGlow)d.removeEventListener("mousemove",r.__landslideGlow);r.__landslideGlow=e=>{r.style.setProperty("--cursor-x",e.clientX+"px");r.style.setProperty("--cursor-y",e.clientY+"px")};d.addEventListener("mousemove",r.__landslideGlow,{passive:true})})()</script>''', height=0)


def risk_bucket(probability: float) -> tuple[str, str, str]:
    if probability < .34:
        return "Low risk", "risk-low", "#4caf7d"
    if probability < .67:
        return "Medium risk", "risk-medium", "#f5b942"
    return "High risk", "risk-high", "#dc5d6b"


def render_prediction(probability: float, threshold: float, dark: bool) -> None:
    label, kind, color = risk_bucket(probability)
    c = theme(dark)
    decision = "At / above model threshold" if probability >= threshold else "Below model threshold"
    
    with st.container(border=True, key="prediction-result"):
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        left, right = st.columns([.85, 1.15], gap="large")
        
        with left:
            st.markdown(f'<div class="risk-summary {kind}"><div class="risk-name">{label.upper()}</div><div class="risk-number">{probability:.1%}</div><div class="risk-copy">Estimated landslide probability</div><span class="decision-pill">{decision} &middot; threshold {threshold:.0%}</span></div>', unsafe_allow_html=True)
        
        with right:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 32, "color": c["text"]}},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 100], "tickcolor": c["text"], "tickfont": {"color": c["muted"]}},
                    "bar": {"color": color, "thickness": .35},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 34], "color": "rgba(76,175,125,.25)"},
                        {"range": [34, 67], "color": "rgba(245,185,66,.22)"},
                        {"range": [67, 100], "color": "rgba(220,93,107,.22)"}
                    ],
                    "threshold": {"line": {"color": "#fff", "width": 3}, "value": threshold * 100}
                }
            ))
            fig.update_layout(
                height=210,
                margin=dict(l=35, r=35, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": c["text"]}
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_feature_importance(items, dark: bool) -> None:
    c = theme(dark)
    names = [FEATURE_NAMES.get(n, n.replace("categorical__", "").replace("numerical__", "").replace("_", " ")) for n, _ in items]
    values = [v * 100 for _, v in items]
    
    with st.container(border=True, key="feature-importance"):
        st.markdown('<div class="section-title">Model Feature Importance</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=values[::-1],
            y=names[::-1],
            orientation="h",
            marker={"color": c["bar"]},
            text=[f"{v:.1f}%" for v in values[::-1]],
            textposition="outside",
            textfont={"color": c["text"]}
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=5, r=58, t=0, b=10),
            xaxis=dict(visible=False, range=[0, max(values) * 1.25]),
            yaxis=dict(tickfont={"color": c["text"], "size": 11}),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Feature importance indicates model usefulness, not causation.")


def render_system_information(metadata: dict, threshold: float, dark: bool) -> None:
    params = metadata.get("best_params") or {}
    cv_f1 = metadata.get("cv_f1")
    
    with st.container(border=True, key="system-information"):
        st.markdown('<div class="section-title">Model / System Information</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-grid"><div class="metric"><span>Model</span><b>{metadata.get("model_name")}</b></div>'
            f'<div class="metric"><span>Features</span><b>{len(metadata.get("features") or [])}</b></div>'
            f'<div class="metric"><span>Decision threshold</span><b>{threshold:.2f}</b></div>'
            f'<div class="metric"><span>CV F1</span><b>{cv_f1:.4f}</b></div></div>',
            unsafe_allow_html=True
        )
        st.caption("Saved Random Forest parameters")
        st.dataframe(
            pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in params.items()]),
            use_container_width=True,
            hide_index=True,
            height=178
        )


def render_current_profile(values: dict, numeric_inputs: dict, dark: bool) -> None:
    c = theme(dark)
    labels = []
    normalized = []
    
    for feature, (low, high, *_) in numeric_inputs.items():
        labels.append(feature.replace("_", " ").replace("3Day", "3 Day"))
        normalized.append((values[feature] - low) / (high - low) * 100 if high != low else 0)
        
    with st.container(border=True, key="current-profile"):
        st.markdown('<div class="section-title">Current Site Profile</div>', unsafe_allow_html=True)
        st.caption("Normalized to the dashboard input ranges; these values are not risk scores.")
        fig = go.Figure(go.Bar(
            x=normalized[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color=c["bar"],
            text=[f"{v:.0f}%" for v in normalized[::-1]],
            textposition="outside",
            textfont={"color": c["text"]}
        ))
        fig.update_layout(
            height=310,
            margin=dict(l=5, r=45, t=0, b=0),
            xaxis=dict(range=[0, 112], visible=False),
            yaxis=dict(tickfont={"color": c["text"], "size": 11}),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_history(history: pd.DataFrame, threshold: float, dark: bool) -> None:
    c = theme(dark)
    count = len(history)
    
    with st.container(border=True, key="prediction-history"):
        st.markdown('<div class="section-title prediction-history-title">Prediction History</div>', unsafe_allow_html=True)
        
        if count == 1:
            value = float(history.iloc[0]["Risk_Probability"])
            st.markdown(f"<div class='metric'><span>Most recent prediction</span><b style='font-size:1.55rem'>{value:.1f}%</b></div>", unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(
                x=[1],
                y=[value],
                mode="markers",
                marker={"size": 16, "color": c["marker"], "line": {"color": "#fff", "width": 2}}
            ))
            layout = dict(height=160, xaxis=dict(range=[.5, 1.5], tickvals=[1], title="Prediction #"))
        else:
            x = list(range(1, count + 1))
            fig = go.Figure(go.Scatter(
                x=x,
                y=history["Risk_Probability"],
                mode="lines+markers",
                line={"color": c["bar"], "width": 3},
                marker={"size": 9, "color": c["marker"]},
                fill="tozeroy",
                fillcolor="rgba(233,122,47,.22)" if dark else "rgba(193,178,255,.20)"
            ))
            layout = dict(height=245, xaxis=dict(title="Prediction #", dtick=1))
            
        layout.update(
            margin=dict(l=35, r=15, t=10, b=30),
            yaxis=dict(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor=c["grid"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=c["plot"],
            font={"color": c["text"]}
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})