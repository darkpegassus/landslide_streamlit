from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "landslide_rfc_model.pkl"
INPUT_COLUMNS = ["Geology_Type", "Land_Use", "Slope_Angle", "Rainfall_3Day", "Effective_Rainfall_mm", "Soil_Saturation", "Vegetation_Cover", "Elevation_m", "Soil_Erosion_Rate", "Pore_Pressure_Ratio", "Seismic_PGA_g"]
CATEGORICAL_OPTIONS = {"Geology_Type": ["Colluvium", "Igneous_Metamorphic", "Residual_Soil", "Sedimentary_Rock", "Weathered_Rock"], "Land_Use": ["Agriculture", "Barren", "Forest", "Urban"]}
# Interface ranges only; values are not scaled or encoded here.
NUMERIC_INPUTS = {
    "Slope_Angle": (0.0, 90.0, 30.0, 1.0, "°", "%.0f"), "Rainfall_3Day": (0.0, 500.0, 100.0, 5.0, "mm", "%.0f"), "Effective_Rainfall_mm": (0.0, 500.0, 80.0, 5.0, "mm", "%.0f"),
    "Soil_Saturation": (0.0, 1.0, 0.5, 0.01, "", "%.2f"), "Vegetation_Cover": (0.0, 1.0, 0.5, 0.01, "", "%.2f"), "Elevation_m": (0.0, 5000.0, 500.0, 10.0, "m", "%.0f"),
    "Soil_Erosion_Rate": (0.0, 20.0, 1.0, 0.1, "", "%.1f"), "Pore_Pressure_Ratio": (0.0, 1.0, 0.5, 0.01, "", "%.2f"), "Seismic_PGA_g": (0.0, 1.0, 0.1, 0.01, "g", "%.2f"),
}
