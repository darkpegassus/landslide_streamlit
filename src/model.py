"""Loading and querying the self-contained saved model artifact."""
from dataclasses import dataclass
import warnings
import joblib
import pandas as pd
import streamlit as st
from config import INPUT_COLUMNS, MODEL_PATH

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
