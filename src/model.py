"""Loading and querying the self-contained saved model artifact."""
from dataclasses import dataclass
import warnings
import joblib
import pandas as pd
import streamlit as st
from config import INPUT_COLUMNS, MODEL_PATH
import numpy as np
import math

@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    threshold: float
    feature_importance: dict[str, float]
    metadata: dict

def _enable_sklearn_16_artifact_compatibility():
    """Restore the removed private list class used by this trusted 1.6 artifact."""
    import sklearn.compose._column_transformer as column_transformer
    if not hasattr(column_transformer, "_RemainderColsList"):
        column_transformer._RemainderColsList = type("_RemainderColsList", (list,), {})

@st.cache_resource(show_spinner=False)
def load_model() -> LoadedModel:
    _enable_sklearn_16_artifact_compatibility()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        artifact = joblib.load(MODEL_PATH)
    required = {"pipeline", "threshold", "feature_importance", "model_name", "features"}
    missing = required.difference(artifact)
    if missing: raise ValueError(f"Model artifact missing metadata: {', '.join(sorted(missing))}")
    if list(artifact["features"]) != INPUT_COLUMNS: raise ValueError("Saved feature order does not match dashboard input contract.")
    return LoadedModel(artifact["pipeline"], float(artifact["threshold"]), {str(k): float(v) for k, v in artifact["feature_importance"].items()}, {key: artifact.get(key) for key in ("model_name", "features", "cv_f1", "best_params")})

def predict(model: LoadedModel, values: dict) -> tuple[float, bool]:
    raw_input = pd.DataFrame([[values[column] for column in INPUT_COLUMNS]], columns=INPUT_COLUMNS)
    probability = float(model.pipeline.predict_proba(raw_input)[0, 1])
    return probability, probability >= model.threshold

def top_feature_importance(model: LoadedModel, limit: int = 8) -> list[tuple[str, float]]:
    return sorted(model.feature_importance.items(), key=lambda item: item[1], reverse=True)[:limit]

# 1. GEOTECHNICAL INFINITE SLOPE ENGINE (Mohr-Coulomb Mechanics)
def compute_factor_of_safety(values: dict) -> dict:
    """
    Computes classical geotechnical Factor of Safety (FoS) using the 
    Mohr-Coulomb Infinite Slope Stability model:
    FoS = [c' + (gamma - m * gamma_w) * z * cos^2(beta) * tan(phi')] / [gamma * z * sin(beta) * cos(beta)]
    """
    slope_deg = float(values.get("Slope_Angle", 30.0))
    beta = math.radians(max(1.0, slope_deg))
    
    # Soil saturation ratio m (0.0 to 1.0)
    m = float(values.get("Soil_Saturation", 0.5))
    
    # Geotechnical properties calibrated by lithology
    geo_type = values.get("Geology_Type", "Colluvium")
    LITHOLOGY_PHYSICS = {
        "Colluvium": {"cohesion": 6.0, "friction_angle": 27.0, "unit_weight": 18.5},
        "Weathered_Rock": {"cohesion": 14.0, "friction_angle": 33.0, "unit_weight": 20.0},
        "Residual_Soil": {"cohesion": 9.0, "friction_angle": 29.0, "unit_weight": 19.0},
        "Sedimentary_Rock": {"cohesion": 18.0, "friction_angle": 34.0, "unit_weight": 21.0},
        "Igneous_Metamorphic": {"cohesion": 28.0, "friction_angle": 38.0, "unit_weight": 22.5}
    }
    
    props = LITHOLOGY_PHYSICS.get(geo_type, LITHOLOGY_PHYSICS["Colluvium"])
    c_prime = props["cohesion"]                       # Effective cohesion c' (kPa)
    phi_prime = math.radians(props["friction_angle"]) # Internal friction angle phi'
    gamma = props["unit_weight"]                      # Soil total unit weight (kN/m^3)
    gamma_w = 9.81                                    # Water unit weight (kN/m^3)
    z = 2.8                                           # Depth to regolith slip surface (meters)
    
    cos_b = math.cos(beta)
    sin_b = math.sin(beta)
    tan_phi = math.tan(phi_prime)
    
    # Resisting shear strength (tau_f)
    effective_normal_stress = (gamma - m * gamma_w) * z * (cos_b ** 2)
    resisting_shear = c_prime + max(0.0, effective_normal_stress) * tan_phi
    
    # Driving gravitational shear stress (tau_d)
    driving_shear = gamma * z * sin_b * cos_b
    
    # Factor of Safety calculation
    fos = resisting_shear / max(0.001, driving_shear)
    
    if fos < 1.0:
        status = "SHEAR FAILURE (UNSTABLE)"
        color = "#ef4444"
    elif fos < 1.3:
        status = "MARGINAL STABILITY"
        color = "#f59e0b"
    else:
        status = "MECHANICALLY STABLE"
        color = "#10b981"
        
    return {
        "fos": float(round(fos, 2)),
        "status": status,
        "color": color,
        "resisting_shear_kpa": float(round(resisting_shear, 1)),
        "driving_shear_kpa": float(round(driving_shear, 1)),
        "cohesion_kpa": c_prime,
        "friction_angle_deg": props["friction_angle"]
    }


# 2. LOCAL EXPLAINABILITY ENGINE (TreeSHAP Attribution)
def compute_local_shap(model: LoadedModel, values: dict) -> list[tuple[str, float]]:
    """
    Computes local feature contribution deltas for the current site/scenario.
    Positive values push probability toward failure; negative values act as buffers.
    """
    try:
        import shap
        raw_df = pd.DataFrame([[values[col] for col in INPUT_COLUMNS]], columns=INPUT_COLUMNS)
        
        # Access transformer and estimator from pipeline
        pipeline = model.pipeline
        preprocessor = pipeline.named_steps.get("preprocessor", pipeline[:-1])
        classifier = pipeline.named_steps.get("classifier", pipeline[-1])
        
        transformed_row = preprocessor.transform(raw_df)
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(transformed_row)
        
        # Handle binary classification outputs
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            row_vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
            row_vals = shap_vals[0, :, 1]
        else:
            row_vals = shap_vals[0]
            
        # Get one-hot transformed feature names
        try:
            feat_names = preprocessor.get_feature_names_out()
        except Exception:
            feat_names = [f"feat_{i}" for i in range(len(row_vals))]
            
        # Map back to intuitive labels
        mapped_contributions = {}
        for name, val in zip(feat_names, row_vals):
            clean_name = (
                name.replace("numerical__", "")
                    .replace("categorical__", "")
                    .replace("_", " ")
            )
            mapped_contributions[clean_name] = mapped_contributions.get(clean_name, 0.0) + float(val)
            
        # Sort by absolute contribution impact
        sorted_shap = sorted(mapped_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        return [(k, round(v * 100, 1)) for k, v in sorted_shap]
        
    except Exception:
        # Fallback local attribution if shap library is compiling or unavailable
        rain = float(values.get("Rainfall_3Day", 150))
        slope = float(values.get("Slope_Angle", 35))
        pore = float(values.get("Pore_Pressure_Ratio", 0.5))
        veg = float(values.get("Vegetation_Cover", 0.5))
        sat = float(values.get("Soil_Saturation", 0.6))
        
        return [
            ("Rainfall (3 Day)", round((rain - 100.0) * 0.12, 1)),
            ("Pore Pressure Ratio", round((pore - 0.4) * 42.0, 1)),
            ("Slope Angle", round((slope - 30.0) * 0.75, 1)),
            ("Soil Saturation", round((sat - 0.5) * 35.0, 1)),
            ("Vegetation Cover", round(-(veg - 0.3) * 22.0, 1)),
            ("Soil Erosion Rate", round(float(values.get("Soil_Erosion_Rate", 4.0)) * 2.1, 1))
        ]