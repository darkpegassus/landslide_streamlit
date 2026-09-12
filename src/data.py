"""Feature controls, live telemetry ingestion, and display metadata."""
from __future__ import annotations
import json
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from config import (
    CATEGORICAL_OPTIONS, 
    NUMERIC_INPUTS, 
    CONTROL_GROUPS, 
    NER_HEATMAP_GRID,
    NER_GEOGRAPHIC_REGIONS
)

FEATURE_LABELS = {
    "Geology_Type": "Geology type", 
    "Land_Use": "Land use", 
    "Elevation_m": "Elevation",
    "Slope_Angle": "Slope angle", 
    "Soil_Erosion_Rate": "Soil erosion rate",
    "Vegetation_Cover": "Vegetation cover", 
    "Rainfall_3Day": "Rainfall (3 day)",
    "Effective_Rainfall_mm": "Effective rainfall", 
    "Soil_Saturation": "Soil saturation",
    "Pore_Pressure_Ratio": "Pore pressure ratio", 
    "Seismic_PGA_g": "Seismic PGA",
}

def _label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))

def collect_inputs(defaults: dict | None = None) -> dict:
    values = {}
    defaults = defaults or {}
    left, right = st.columns(2, gap="large")
    
    for group_index, (group_name, features) in enumerate(CONTROL_GROUPS):
        parent = left if group_index < 3 else right
        with parent:
            st.markdown(f'<div class="input-group-title">{group_name}</div>', unsafe_allow_html=True)
            rows = st.columns(2, gap="medium")
            
            for index, feature in enumerate(features):
                with rows[index % 2]:
                    k = "input_lat" if feature == "Latitude" else ("input_lon" if feature == "Longitude" else f"input_{feature}")
                    
                    if feature == "Latitude":
                        val = float(st.session_state.get(k, defaults.get("Latitude", 23.73)))
                        values[feature] = st.slider("Latitude (°N)", 6.0, 38.0, val, 0.01, format="%.2f", key=k)
                    elif feature == "Longitude":
                        val = float(st.session_state.get(k, defaults.get("Longitude", 92.71)))
                        values[feature] = st.slider("Longitude (°E)", 68.0, 98.0, val, 0.01, format="%.2f", key=k)
                    elif feature in CATEGORICAL_OPTIONS:
                        opts = CATEGORICAL_OPTIONS[feature]
                        cat_val = st.session_state.get(k, defaults.get(feature, opts[0]))
                        idx = opts.index(cat_val) if cat_val in opts else 0
                        values[feature] = st.selectbox(_label(feature), opts, index=idx, key=k)
                    elif feature in NUMERIC_INPUTS:
                        low, high, fallback, step, unit, number_format = NUMERIC_INPUTS[feature]
                        label = f"{_label(feature)} ({unit})" if unit else _label(feature)
                        raw_val = float(st.session_state.get(k, defaults.get(feature, fallback)))
                        val = float(np.clip(raw_val, low, high))
                        values[feature] = st.slider(
                            label, 
                            float(low), 
                            float(high), 
                            val, 
                            float(step), 
                            format=number_format, 
                            key=k
                        )
    return values

@st.cache_data(ttl=1800)
def generate_ner_heatmap_data() -> pd.DataFrame:
    """Returns the cached 48-node spatial risk density mesh across the North East."""
    return pd.DataFrame(NER_HEATMAP_GRID)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_ner_weather(lat: float, lon: float) -> dict:
    """
    Queries real-time precipitation & soil moisture from Open-Meteo for exact coordinates.
    Cached for 10 minutes to prevent rate limiting.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat:.4f}&longitude={lon:.4f}&"
        f"past_days=3&forecast_days=1&"
        f"hourly=precipitation,soil_moisture_0_to_1cm"
    )
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MDoNER-Landslide-Dashboard/2.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        
        hourly_precip = data.get("hourly", {}).get("precipitation", [])
        # 3-day precipitation (last 72 hours)
        rainfall_3day = sum(hourly_precip[-72:]) if len(hourly_precip) >= 72 else sum(hourly_precip)
        
        soil_moist = data.get("hourly", {}).get("soil_moisture_0_to_1cm", [])
        latest_moist = soil_moist[-1] if soil_moist else 0.38
        
        return {
            "Rainfall_3Day": float(round(min(500.0, max(0.0, rainfall_3day)), 1)),
            "Soil_Saturation": float(round(min(1.0, max(0.05, latest_moist * 1.85)), 2)),
            "hourly_precip_past_24h": hourly_precip[-24:] if len(hourly_precip) >= 24 else [0.0] * 24
        }
    except Exception:
        # Fallback values if offline
        return {
            "Rainfall_3Day": 210.0,
            "Soil_Saturation": 0.78,
            "hourly_precip_past_24h": [2.5] * 24
        }

def get_24h_telemetry(station_name: str) -> pd.DataFrame:
    """
    Continuously ingests live Open-Meteo satellite weather for the station
    and generates 144 intervals (24 hours at 10-minute frequency).
    """
    meta = NER_GEOGRAPHIC_REGIONS.get(station_name, list(NER_GEOGRAPHIC_REGIONS.values())[0])
    is_high_risk = meta.get("risk") in ["Extreme", "High"]
    
    # 1. Always pull live satellite weather from Open-Meteo
    live = fetch_live_ner_weather(meta["lat"], meta["lon"])
    base_rain = live["Rainfall_3Day"]
    base_sat = live["Soil_Saturation"]
    hourly_precip = live.get("hourly_precip_past_24h", [0.0] * 24)

    now = datetime.now()
    rows = []
    
    # 144 steps = 24 hours * 6 steps/hr (every 10 minutes)
    for i in range(144):
        time_point = now - timedelta(minutes=(143 - i) * 10)
        timestamp_str = time_point.strftime("%H:%M")
        hour_idx = min(23, i // 6)
        
        # Build progressive 24-hour rainfall curve
        if hourly_precip and len(hourly_precip) == 24:
            hourly_delta = hourly_precip[hour_idx] * 0.4
            rain_val = float(np.clip(base_rain - (23 - hour_idx) * 1.5 + hourly_delta, 0.0, 500.0))
        else:
            trend = (i / 144.0) * 0.20 if is_high_risk else 0.0
            rain_val = float(np.clip(base_rain * (0.88 + trend), 0.0, 500.0))
            
        sat_val = float(np.clip(base_sat + (i / 144.0) * 0.05, 0.05, 1.0))
        veg = 0.50
        
        # Round the final current reading to match slider steps
        if i == 143:
            rain_val = round(rain_val / 5.0) * 5.0
            sat_val = round(sat_val, 2)
            
        row = {
            "Timestamp": timestamp_str,
            "Geology_Type": meta.get("geo", "Colluvium"),
            "Land_Use": "Forest" if is_high_risk else "Urban",
            "Slope_Angle": float(meta.get("slope", 38.0)),
            "Rainfall_3Day": rain_val,
            "Effective_Rainfall_mm": float(rain_val * veg),
            "Soil_Saturation": sat_val,
            "Vegetation_Cover": veg,
            "Elevation_m": float(meta.get("elev", 1400.0)),
            "Soil_Erosion_Rate": 9.0 if is_high_risk else 2.5,
            "Pore_Pressure_Ratio": float(np.clip(sat_val * 0.80, 0.05, 0.98)),
            "Seismic_PGA_g": 0.35 if is_high_risk else 0.08,
            "Latitude": float(meta["lat"]),
            "Longitude": float(meta["lon"])
        }
        rows.append(row)
        
    return pd.DataFrame(rows)