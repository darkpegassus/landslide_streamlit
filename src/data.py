"""Feature controls and display metadata for the dashboard."""
import streamlit as st

from config import CATEGORICAL_OPTIONS, NUMERIC_INPUTS

FEATURE_LABELS = {
    "Geology_Type": "Geology type", "Land_Use": "Land use", "Elevation_m": "Elevation",
    "Slope_Angle": "Slope angle", "Soil_Erosion_Rate": "Soil erosion rate",
    "Vegetation_Cover": "Vegetation cover", "Rainfall_3Day": "Rainfall (3 day)",
    "Effective_Rainfall_mm": "Effective rainfall", "Soil_Saturation": "Soil saturation",
    "Pore_Pressure_Ratio": "Pore pressure ratio", "Seismic_PGA_g": "Seismic PGA",
}

CONTROL_GROUPS = (
    ("Site Location", ("Latitude", "Longitude")),
    ("Site Characteristics", ("Geology_Type", "Land_Use", "Elevation_m")),
    ("Seismic", ("Seismic_PGA_g",)),
    ("Terrain", ("Slope_Angle", "Soil_Erosion_Rate", "Vegetation_Cover")),
    ("Hydrology", ("Rainfall_3Day", "Effective_Rainfall_mm", "Soil_Saturation", "Pore_Pressure_Ratio")),
)

def _label(feature: str) -> str:
    return FEATURE_LABELS[feature]

def _render_control(feature: str, column) -> tuple[str, object]:
    with column:
        if feature == "Latitude":
            return feature, st.slider("Latitude", 6.0, 38.0, 20.59, 0.01, key="input_lat")
        if feature == "Longitude":
            return feature, st.slider("Longitude", 68.0, 98.0, 78.96, 0.01, key="input_lon")
        if feature in CATEGORICAL_OPTIONS:
            return feature, st.selectbox(_label(feature), CATEGORICAL_OPTIONS[feature], key=f"input_{feature}")
        low, high, default, step, unit, number_format = NUMERIC_INPUTS[feature]
        label = f"{_label(feature)} ({unit})" if unit else _label(feature)
        return feature, st.slider(label, low, high, default, step, format=number_format, key=f"input_{feature}")


def collect_inputs(defaults: dict = None) -> dict:
    values = {}
    left, right = st.columns(2, gap="large")
    for group_index, (group_name, features) in enumerate(CONTROL_GROUPS):
        parent = left if group_index < 3 else right
        with parent:
            st.markdown(f'<div class="input-group-title">{group_name}</div>', unsafe_allow_html=True)
            rows = st.columns(2, gap="medium")
            for index, feature in enumerate(features):
                with rows[index % 2]:
                    # Determine default value
                    d_val = defaults.get(feature) if defaults else None
                    
                    if feature == "Latitude":
                        values[feature] = st.slider("Latitude", 6.0, 38.0, d_val or 20.59, 0.01, key="input_lat")
                    elif feature == "Longitude":
                        values[feature] = st.slider("Longitude", 68.0, 98.0, d_val or 78.96, 0.01, key="input_lon")
                    elif feature in CATEGORICAL_OPTIONS:
                        opts = CATEGORICAL_OPTIONS[feature]
                        idx = opts.index(d_val) if d_val in opts else 0
                        values[feature] = st.selectbox(_label(feature), opts, index=idx, key=f"input_{feature}")
                    else:
                        low, high, default, step, unit, number_format = NUMERIC_INPUTS[feature]
                        label = f"{_label(feature)} ({unit})" if unit else _label(feature)
                        values[feature] = st.slider(label, float(low), float(high), float(d_val if d_val is not None else default), float(step), format=number_format, key=f"input_{feature}")
    return values