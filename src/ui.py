"""Theme-aware visual components for the landslide dashboard."""
from __future__ import annotations
from config import (
    MONITORING_STATIONS, 
    NER_GEOGRAPHIC_REGIONS, 
    SOP_LEVELS, 
    HIGHWAY_CORRIDORS, 
    HIGHWAY_POLYLINE_COORDINATES,
    CAP_LANGUAGES,
    MULTILINGUAL_CAP_ALERTS
)
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

FEATURE_NAMES = {"numerical__Slope_Angle": "Slope angle", "numerical__Vegetation_Cover": "Vegetation cover", "numerical__Soil_Erosion_Rate": "Soil erosion rate", "categorical__Geology_Type_Colluvium": "Geology type: Colluvium", "numerical__Seismic_PGA_g": "Seismic PGA", "numerical__Pore_Pressure_Ratio": "Pore pressure ratio", "numerical__Soil_Saturation": "Soil saturation", "numerical__Rainfall_3Day": "Rainfall (3 day)"}


def theme(dark: bool) -> dict[str, str]:
    return ({"text": "#f4e8dc", "muted": "#cbb8aa", "accent": "#f0954f", "grid": "rgba(240,149,79,.24)", "plot": "rgba(16,10,8,.48)", "bar": "#e98945", "marker": "#ffc08d"} 
            if dark else 
            {"text": "#1a1625", "muted": "#4a4458", "accent": "#42327d", "grid": "rgba(36,27,66,0.1)", "plot": "rgba(235,229,250,0.5)", "bar": "#5c49a0", "marker": "#b03a5c"})


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
        "METRIC": "rgba(85,39,21,0.48)" if dark else "rgba(210,200,250,0.4)", # Darker boxes in light mode
        "CARD_BORDER": "#f0954f" if dark else "#8d7ddd", # Clearer borders in light mode
        "TEXT": c["text"], 
        "MUTED": c["muted"], 
        "ACCENT": c["accent"], 
        "GRID": c["grid"], 
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
        
        /* 1. BROAD SELECTOR FOR 3D EFFECT */
        div[class*="st-key-current-parameters"],
        div[class*="st-key-prediction-result"],
        div[class*="st-key-feature-importance"],      /* Restored */
        div[class*="st-key-system-information"],       /* Restored */
        div[class*="st-key-current-profile"],
        div[class*="st-key-live-history-trend"],
        div[class*="st-key-user-sim-history"],
        div[class*="st-key-cap-broadcast-container"] {
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            transform: perspective(950px) translateZ(0);
        }
        
        /* 2. HOVER SELECTOR */
        div[class*="st-key-current-parameters"]:hover,
        div[class*="st-key-prediction-result"]:hover,
        div[class*="st-key-feature-importance"]:hover, /* Restored */
        div[class*="st-key-system-information"]:hover,  /* Restored */
        div[class*="st-key-current-profile"]:hover,
        div[class*="st-key-live-history-trend"]:hover,
        div[class*="st-key-user-sim-history"]:hover,
        div[class*="st-key-cap-broadcast-container"]:hover {
            border-color: __HOVER__ !important;
            transform: translateY(-6px) perspective(950px) rotateX(2deg) rotateY(-0.8deg);
            box-shadow: __SHADOW__;
        }
        div[data-testid="stNotification"] p {
            color: inherit !important;
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
            min-height: 3rem;
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
            width: 84%;
            font-size: 0.01rem;
            min-height: 1.8rem;
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


def render_prediction(probability: float, threshold: float, dark: bool, active_station: str = "Aizawl") -> None:
    label, kind, color = risk_bucket(probability)
    c = theme(dark)
    decision = "At / above model threshold" if probability >= threshold else "Below model threshold"
    
    # Identify applicable highway corridor
    corridor_info = HIGHWAY_CORRIDORS.get(active_station, {"highway": "Regional Arterial Link", "lifeline_for": "Regional Civil Transit"})
    
    # Determine Active SOP Level
    active_sop = SOP_LEVELS[0]
    for sop in SOP_LEVELS:
        if sop["range"][0] <= probability <= sop["range"][1]:
            active_sop = sop
            break
    if probability > 0.85:
        active_sop = SOP_LEVELS[3]

    with st.container(border=True, key="prediction-result"):
        st.markdown('<div class="section-title">Incident Command Center & Threat Assessment</div>', unsafe_allow_html=True)
        
        col_summary, col_gauge, col_matrix = st.columns([0.85, 1.05, 1.3], gap="medium")
        
        # 1. Left: Risk Summary Card
        with col_summary:
            st.markdown(
                f'<div class="risk-summary {kind}">'
                f'<div class="risk-name">{label.upper()}</div>'
                f'<div class="risk-number">{probability:.1%}</div>'
                f'<div class="risk-copy">Landslide Probability Index</div>'
                f'<span class="decision-pill">{decision} &middot; threshold {threshold:.0%}</span>'
                f'</div>', 
                unsafe_allow_html=True
            )
        
        # 2. Middle: Speedometer Gauge
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 30, "color": c["text"]}},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 100], "tickcolor": c["text"], "tickfont": {"color": c["muted"], "size": 10}},
                    "bar": {"color": active_sop["color"], "thickness": 0.35},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(16, 185, 129, 0.22)"},
                        {"range": [35, 65], "color": "rgba(245, 158, 11, 0.22)"},
                        {"range": [65, 85], "color": "rgba(249, 115, 22, 0.25)"},
                        {"range": [85, 100], "color": "rgba(239, 68, 68, 0.28)"}
                    ]}
            ))
            fig.update_layout(
                height=195,
                margin=dict(l=25, r=25, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": c["text"]}
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # 3. Right: Dynamic Incident Command Action Matrix
        with col_matrix:
            matrix_html = (
                f'<div style="background: rgba(0,0,0,0.06); border: 1.5px solid {active_sop["color"]}; '
                f'border-radius: 14px; padding: 12px 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">'
                f'<span style="font-size: 0.72rem; font-weight:800; letter-spacing:0.08em; color: {active_sop["color"]}; text-transform:uppercase;">'
                f'INCIDENT COMMAND DIRECTIVE</span>'
                f'<span style="background:{active_sop["color"]}; color:#000; font-weight:800; font-size:0.65rem; padding: 2px 7px; border-radius: 999px;">'
                f'{active_sop["badge"]}</span>'
                f'</div>'
                f'<div style="font-family: \'Barlow Condensed\', sans-serif; font-size: 1.25rem; font-weight:700; color:{c["text"]}; margin-bottom:2px;">'
                f'{corridor_info["highway"]}</div>'
                f'<div style="font-size:0.75rem; color:{c["muted"]}; margin-bottom:8px;">{corridor_info["lifeline_for"]}</div>'
                f'<div style="font-size:0.82rem; font-weight:600; line-height:1.35; color:{c["text"]}; background:rgba(255,255,255,0.04); padding:8px; border-radius:8px; border-left:3px solid {active_sop["color"]};">'
                f'👉 {active_sop["action"]}</div>'
                f'</div>'
            )
            st.markdown(matrix_html, unsafe_allow_html=True)

def render_model_insights_dual(
    shap_items: list[tuple[str, float]], 
    fos_data: dict, 
    ml_probability: float, 
    dark: bool
) -> None:
    """
    Renders an interactive switch between:
    1. Local TreeSHAP Explainability (Why this specific slope is failing)
    2. Geotechnical Infinite Slope Engine (Mohr-Coulomb Factor of Safety)
    """
    c = theme(dark)
    
    with st.container(border=True, key="feature-importance"):
        st.markdown('<div class="section-title">Diagnostic Engine: XAI & Geotechnical Mechanics</div>', unsafe_allow_html=True)
        
        engine_mode = st.radio(
            "Select Diagnostic Mode",
            ["🔬 Local TreeSHAP (XAI Feature Attribution)", "⚙️ Geotechnical Physics (Factor of Safety)"],
            horizontal=True,
            label_visibility="collapsed",
            key="insight_engine_radio"
        )
        
        if "TreeSHAP" in engine_mode:
            st.caption("Local feature attribution for the active station: indicates which parameters drive failure versus mitigate it.")
            
            names = [item[0] for item in shap_items]
            values = [item[1] for item in shap_items]
            bar_colors = ["#ef4444" if v >= 0 else "#10b981" for v in values]
            
            fig = go.Figure(go.Bar(
                x=values[::-1],
                y=names[::-1],
                orientation="h",
                marker=dict(color=bar_colors[::-1]),
                text=[f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%" for v in values[::-1]],
                textposition="outside",
                textfont=dict(color=c["text"], size=11)
            ))
            
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor=c["grid"], zeroline=True, zerolinecolor=c["text"]),
                yaxis=dict(tickfont=dict(color=c["text"], size=11)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
        else:
            # Geotechnical Infinite Slope Mechanics
            fos_val = fos_data["fos"]
            status_color = fos_data["color"]
            
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.06); border:1.5px solid {status_color}; border-radius:12px; padding:12px 16px; margin-bottom:12px;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div><span style="font-size:0.75rem; font-weight:700; color:{c["muted"]};">MOHR-COULOMB FACTOR OF SAFETY</span>'
                f'<div style="font-size:2.2rem; font-weight:800; color:{status_color}; line-height:1;">{fos_val:.2f}</div></div>'
                f'<span style="background:{status_color}; color:#000; font-weight:800; font-size:0.78rem; padding:4px 10px; border-radius:999px;">'
                f'{fos_data["status"]}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Physics breakdown metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric"><span>Resisting Shear</span><b>{fos_data["resisting_shear_kpa"]} kPa</b></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric"><span>Driving Stress</span><b>{fos_data["driving_shear_kpa"]} kPa</b></div>', unsafe_allow_html=True)
            with m3:
                # Convergence analysis between AI and Physics
                if ml_probability >= 0.65 and fos_val < 1.0:
                    badge_text = "TIER-1 CONVERGENCE"
                    b_color = "#ef4444"
                elif ml_probability < 0.35 and fos_val >= 1.3:
                    badge_text = "DUAL-STABLE"
                    b_color = "#10b981"
                else:
                    badge_text = "HYBRID WATCH"
                    b_color = "#f59e0b"
                st.markdown(f'<div class="metric"><span style="color:{b_color}; font-weight:800;">Validation</span><b>{badge_text}</b></div>', unsafe_allow_html=True)
                
            st.caption("Infinite Slope Stability Model: FoS < 1.0 indicates physical shear failure under current gravitational and pore pressure loads.")

def render_system_information(metadata: dict, threshold: float, dark: bool) -> None:
    params = metadata.get("best_params") or {}
    cv_f1 = metadata.get("cv_f1")
    
    # Parameters to hide from the table
    EXCLUDED_PARAMS = {
        "max_features",
    }
    
    # Filter out the unwanted parameters
    filtered_table_data = [
        {"Parameter": k, "Value": str(v)} 
        for k, v in params.items() 
        if k not in EXCLUDED_PARAMS
    ]
    
    with st.container(border=True, key="system-information"):
        st.markdown('<div class="section-title">Model / System Information</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-grid">'
            f'<div class="metric"><span>Model</span><b>{metadata.get("model_name")}</b></div>'
            f'<div class="metric"><span>Features</span><b>{len(metadata.get("features") or [])}</b></div>'
            f'<div class="metric"><span>Decision threshold</span><b>{threshold:.2f}</b></div>'
            f'<div class="metric"><span>CV F1</span><b>{cv_f1:.4f}</b></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.caption("Saved Random Forest parameters")
        st.dataframe(
            pd.DataFrame(filtered_table_data),
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


def render_history(history: pd.DataFrame, threshold: float, dark: bool, key: str = "prediction-history") -> None:
    c = theme(dark)
    count = len(history)
    line_color = c["bar"]
    marker_color = "#ffffff" if dark else "#241b42"
    text_color = c["text"]
    grid_color = c["grid"]
    
    with st.container(border=True, key=key):
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
            layout = dict(height=180, xaxis=dict(range=[0.5, 1.5], tickvals=[1], title="Prediction #"))
        else:
            # Check if timestamps are available (used in the 24-hour 144-point series)
            if "Timestamp" in history.columns:
                x_vals = history["Timestamp"].tolist()
                x_title = "Time (Last 24 Hours, 10-min interval)"
                # Show tick marks roughly every 3 hours (every 18 points) to avoid visual clutter
                tick_step = 18
                tick_indices = list(range(0, count, tick_step))
                if (count - 1) not in tick_indices:
                    tick_indices.append(count - 1)
                tick_vals = [x_vals[idx] for idx in tick_indices]
            else:
                x_vals = list(range(1, count + 1))
                x_title = "Simulation Run #"
                tick_vals = x_vals
                
            fig = go.Figure(go.Scatter(
                x=x_vals,
                y=history["Risk_Probability"],
                mode="lines",
                line=dict(color=c["bar"], width=2.5),
                fill="tozeroy",
                fillcolor="rgba(233,122,47,.20)" if dark else "rgba(193,178,255,.25)",
                hovertemplate="Time: %{x}<br>Risk: <b>%{y:.1f}%</b><extra></extra>"
            ))
            
            layout = dict(
                height=260,
                xaxis=dict(
                    title=x_title,
                    tickvals=tick_vals,
                    tickfont=dict(color=text_color, size=10),
                    gridcolor=grid_color
                )
            )
            
        layout.update(
            margin=dict(l=35, r=15, t=10, b=30),
            yaxis=dict(
                range=[0, 100], 
                ticksuffix="%", 
                showgrid=True, 
                gridcolor=grid_color,
                tickfont=dict(color=text_color)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=c["plot"],
            font=dict(color=text_color, family="Poppins, sans-serif")
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_interactive_map(selected_station: str, dark: bool, current_risk_prob: float = 0.5):
    c = theme(dark)
    is_v6 = hasattr(go, "Scattermap")
    scatter_type = go.Scattermap if is_v6 else go.Scattermapbox
    density_type = go.Densitymap if is_v6 else go.Densitymapbox
    layout_key = "map" if is_v6 else "mapbox"

    fig = go.Figure()

    # 1. ADD CONNECTED HIGHWAY LIFELINE POLYLINES
    for hwy_key, hwy_data in HIGHWAY_POLYLINE_COORDINATES.items():
        coords = hwy_data["coordinates"]
        lats = [pt[0] for pt in coords]
        lons = [pt[1] for pt in coords]
        
        # Color line based on risk if this highway belongs to the selected station
        if hwy_data["associated_station"] == selected_station:
            if current_risk_prob < 0.35:
                line_color = "#10b981"  # Emerald Green (Safe)
                line_width = 4.5
                status_text = "STATUS: CLEAR / OPEN"
            elif current_risk_prob < 0.65:
                line_color = "#f59e0b"  # Amber (Watch)
                line_width = 5.0
                status_text = "STATUS: BRO STANDBY"
            elif current_risk_prob < 0.85:
                line_color = "#f97316"  # Orange (Warning)
                line_width = 5.5
                status_text = "STATUS: DIVERSION ACTIVE"
            else:
                line_color = "#ef4444"  # Neon Crimson Red (Emergency Shutdown)
                line_width = 6.5
                status_text = "STATUS: MANDATORY SEC 144 SHUTDOWN"
        else:
            # Static fallback for unselected highways
            line_color = "#001944"
            line_width = 3.0
            status_text = "STATUS: ROUTINE PATROL"

        fig.add_trace(scatter_type(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(color=line_color, width=line_width),
            hoverinfo="text",
            hovertext=f"<b>{hwy_key}</b><br>{status_text}<br>Chokepoints: {hwy_data['chokepoints']}",
            name=hwy_key
        ))

    # 2. ADD RAINBOW HAZARD DENSITY HEATMAP
    WEIGHTS = {"Extreme": 1.0, "High": 0.78, "Medium": 0.48, "Low": 0.18}
    h_lats, h_lons, h_weights = [], [], []
    for d in MONITORING_STATIONS.values():
        h_lats.append(d['lat'])
        h_lons.append(d['lon'])
        h_weights.append(WEIGHTS.get(d.get('risk', 'Low'), 0.2))

    fig.add_trace(density_type(
        lat=h_lats,
        lon=h_lons,
        z=h_weights,
        radius=55,
        opacity=0.65,
        colorscale=[
            [0.00, "rgba(80, 0, 100, 0.0)"],
            [0.18, "rgba(90, 0, 140, 0.70)"],
            [0.35, "rgba(0, 50, 255, 0.80)"],
            [0.52, "rgba(0, 220, 255, 0.85)"],
            [0.68, "rgba(50, 230, 50, 0.88)"],
            [0.82, "rgba(255, 230, 0, 0.92)"],
            [0.92, "rgba(255, 120, 0, 0.95)"],
            [1.00, "rgba(255, 0, 0, 1.00)"]
        ],
        colorbar=dict(
            title=dict(
                text="Hazard Index",
                font=dict(color=c["text"], size=10, family="Poppins, sans-serif")
            ),
            tickvals=[0.2, 0.48, 0.78, 1.0],
            ticktext=["Low", "Moderate", "High", "Critical"],
            tickfont=dict(color=c["text"], size=9),
            thickness=9,
            len=0.70,
            # FLOATING HUD POSITIONING (Adjust 'x' freely without affecting map shape):
            x=1,                  
            xanchor="right",      
            y=0.5,               
            yanchor="middle",         
            bgcolor="rgba(0,0,0,0.25)",      
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        showscale=True,
        hoverinfo="none",
        name="Hazard Density"
    ))

    # 3. STATION PIN MARKERS
    m_lats, m_lons, m_names, m_colors, m_sizes, m_hovers = [], [], [], [], [], []
    for name, data in MONITORING_STATIONS.items():
        m_lats.append(data['lat'])
        m_lons.append(data['lon'])
        m_names.append(name)
        risk_lvl = data.get('risk', 'Low')
        
        if name == selected_station:
            m_colors.append("#a600ff" if dark else "#2a00ff")
            m_sizes.append(22)
        else:
            m_colors.append("#dc2626" if risk_lvl in ["Extreme", "High"] else "#16a34a")
            m_sizes.append(12)
            
        m_hovers.append(f"<b>{name}</b><br>Hazard Level: <b>{risk_lvl}</b><br><i>Click to inspect telemetry</i>")

    fig.add_trace(scatter_type(
        lat=m_lats,
        lon=m_lons,
        mode='markers',
        marker=dict(size=m_sizes, color=m_colors, opacity=0.9),
        text=m_hovers,
        hoverinfo="text",
        customdata=m_names,
        name="Station Pins"
    ))

    # Center camera on selected station
    sel_data = MONITORING_STATIONS.get(selected_station, list(MONITORING_STATIONS.values())[0])
    
    fig.update_layout(
        **{layout_key: dict(
            # THIS PREVENTS PLOTLY FROM SQUEEZING/RESHAPING THE MAP:
            domain=dict(x=[0, 1], y=[0, 1]),
            
            style="white-bg",
            layers=[{
                "below": 'traces',
                "sourcetype": "raster",
                "source": ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"]
            }],
            center=dict(lat=sel_data['lat'], lon=sel_data['lon']),
            zoom=4.5
        )},
        margin={"r":0, "t":0, "l":0, "b":0},
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    with st.container(border=True, key="physical-map-card"):
        st.markdown('<div class="section-title">Hazard Heatmap & Lifeline Highway Corridors</div>', unsafe_allow_html=True)
        
        evt = st.plotly_chart(
            fig, 
            use_container_width=True, 
            on_select="rerun", 
            key="main_map_plot", 
            config={'displayModeBar': False, 'scrollZoom': True}
        )

        if evt and "selection" in evt and "points" in evt["selection"] and evt["selection"]["points"]:
            for pt in evt["selection"]["points"]:
                clicked_name = pt.get("customdata")
                if clicked_name and clicked_name in MONITORING_STATIONS:
                    return clicked_name

    return selected_station

def render_multilingual_cap_suite(
    station_name: str, 
    probability: float, 
    dark: bool
) -> None:
    """
    Renders an NDMA/SACHET-compliant Common Alerting Protocol (CAP v1.2)
    public broadcast suite in 6 North-Eastern regional languages.
    """
    c = theme(dark)
    now_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Classify Alert Category
    if probability >= 0.75:
        tier_key = "Extreme"
        badge_color = "#ef4444"
        badge_text = "CAP PRIORITY 1 · IMMEDIATE EVACUATION"
    elif probability >= 0.40:
        tier_key = "High"
        badge_color = "#f59e0b"
        badge_text = "CAP PRIORITY 2 · SEVERE THREAT"
    else:
        tier_key = "Low"
        badge_color = "#10b981"
        badge_text = "CAP PRIORITY 4 · ROUTINE BULLETIN"
        
    cap_data = MULTILINGUAL_CAP_ALERTS[tier_key]
    corridor_data = HIGHWAY_CORRIDORS.get(station_name, {"highway": "Regional Lifeline"})

    with st.container(border=True, key="cap-broadcast-container"):
        st.markdown(
            '<div class="section-title">Emergency Warning & CAP Public Broadcast (NDMA SACHET Protocol)</div>', 
            unsafe_allow_html=True
        )
        
        # Language Selector Bar
        col_lang, col_stat = st.columns([1.8, 1.2])
        with col_lang:
            selected_lang = st.selectbox(
                "Broadcast Language (North-Eastern Region)",
                list(CAP_LANGUAGES.keys()),
                index=0,
                key="cap_lang_selector"
            )
        with col_stat:
            st.markdown(
                f'<div style="text-align:right; padding-top:25px;">'
                f'<span style="background:{badge_color}; color:#000; font-weight:800; font-size:0.72rem; padding:4px 10px; border-radius:999px;">'
                f'{badge_text}</span></div>', 
                unsafe_allow_html=True
            )

        lang_code = CAP_LANGUAGES[selected_lang]
        msg = cap_data["templates"][selected_lang]

        # STYLIZED PHONE CELL BROADCAST CARD
        cell_card_html = f"""
        <div style="background: rgba(0,0,0,0.12); border: 2px solid {badge_color}; border-radius: 16px; padding: 16px 20px; margin: 12px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; margin-bottom: 12px;">
                <span style="font-size:0.75rem; font-weight:800; letter-spacing:0.1em; color:{badge_color};">
                    🚨 CELL BROADCAST ADVISORY · GOVT. OF INDIA (NDMA / SDMA)
                </span>
                <span style="font-size:0.72rem; color:{c['muted']}; font-family:monospace;">{now_utc}</span>
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color:{c['text']}; margin-bottom: 6px;">
                {msg['headline']}
            </div>
            <div style="font-size: 0.85rem; color:{c['muted']}; margin-bottom: 10px;">
                <b>Target Area:</b> {station_name} Sector &middot; Corridor: <b>{corridor_data['highway']}</b> &middot; [BCP-47: <code>{lang_code}</code>]
            </div>
            <div style="font-size: 0.92rem; font-weight: 500; line-height: 1.45; color:{c['text']}; background: rgba(255,255,255,0.04); padding: 12px; border-radius: 8px; border-left: 4px solid {badge_color};">
                {msg['instruction']}
            </div>
        </div>
        """
        st.markdown(cell_card_html, unsafe_allow_html=True)

        # TECHNICAL EXPORT: RAW CAP XML PAYLOAD
        with st.expander("📄 View Machine-Readable CAP v1.2 XML Document (NDMA SACHET Standard)"):
            cap_xml_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>MDoNER-NDMA-{station_name.replace(' ', '_')}-{int(datetime.now().timestamp())}</identifier>
  <sender>ndma-sachet-dispatcher@gov.in</sender>
  <sent>{now_utc}</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>{lang_code}</language>
    <category>Geo</category>
    <event>Landslide / Slope Failure</event>
    <urgency>{cap_data['urgency']}</urgency>
    <severity>{cap_data['severity']}</severity>
    <certainty>{cap_data['certainty']}</certainty>
    <headline>{msg['headline']}</headline>
    <description>AI telemetry indicates {probability:.1%} probability of imminent slope destabilization along {corridor_data['highway']}.</description>
    <instruction>{msg['instruction']}</instruction>
    <area>
      <areaDesc>{station_name} District and Lifeline Corridors</areaDesc>
    </area>
  </info>
</alert>"""
            st.code(cap_xml_payload, language="xml")
            st.download_button(
                label="⬇️ Download Signed CAP XML Payload",
                data=cap_xml_payload,
                file_name=f"CAP_Alert_{station_name}_{lang_code}.xml",
                mime="application/xml",
                key="btn_download_cap_xml"
            )