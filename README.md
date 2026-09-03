# Landslide Risk Prediction Dashboard

A Streamlit web application that estimates the probability of a landslide from site conditions. It provides an interactive, dark-themed dashboard for entering environmental inputs, viewing a risk classification, and comparing predictions during the current browser session.

> **Important:** This project is a demonstration and decision-support tool. Its output must not be used as the sole basis for public-safety decisions, evacuation orders, construction approval, or emergency response. Consult qualified geotechnical and disaster-management professionals for real-world assessments.

## Features

- Interactive controls for rainfall, slope angle, soil saturation, and vegetation cover.
- Model-based risk probability and Low / Medium / High classification.
- Risk gauge and normalized radar chart for the selected inputs.
- Session-only prediction history, including a probability trend chart.
- Bundled trained model and feature scaler, loaded with `joblib`.

## Inputs and ranges

| Input | Description | Allowed range | Default |
| --- | --- | ---: | ---: |
| Rainfall | Rainfall at the site | 0–300 mm | 100 mm |
| Slope angle | Inclination of the slope | 5–80° | 30° |
| Soil saturation | Relative soil-water saturation | 0.10–1.00 | 0.50 |
| Vegetation cover | Relative vegetation coverage | 0.00–1.00 | 0.50 |

The application scales these features with `landslide_scaler4.pkl` before passing them to `landslide_model4.pkl`.

## Risk levels

The dashboard uses the model's predicted positive-class probability:

| Probability | Displayed level |
| ---: | --- |
| Less than 34% | Low |
| 34% to less than 67% | Medium |
| 67% or more | High |

These display thresholds are application rules; they do not replace a validated local hazard-warning protocol.

## Requirements

- Python 3.14 or later (as specified in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) recommended for dependency management, or `pip`

Key libraries include Streamlit, scikit-learn, pandas, NumPy, Plotly, and joblib. Exact locked versions are recorded in `uv.lock`.

## Run locally

From this directory (`landslide_streamlit`):

```powershell
uv sync
uv run streamlit run src/app.py
```

Alternatively, with a virtual environment and pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install streamlit scikit-learn pandas numpy plotly joblib
streamlit run src/app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`.

## Project layout

```text
landslide_streamlit/
├── src/
│   └── app.py                    # Streamlit UI and prediction workflow
├── landslide_model4.pkl          # Trained prediction model
├── landslide_scaler4.pkl         # Feature scaler used by the model
├── pyproject.toml                # Project metadata and dependencies
└── uv.lock                       # Locked dependency versions
```

## How it works

1. The sidebar collects the four site-condition values.
2. When **Predict Landslide Risk** is selected, the values are assembled in this fixed order: `Rainfall_mm`, `Slope_Angle`, `Soil_Saturation`, `Vegetation_Cover`.
3. The saved scaler transforms the values.
4. The saved model produces a class and, where supported, a probability. The positive-class probability drives the risk display.
5. The result is added to in-memory Streamlit session history. **Reset History** clears that history; it is not persisted between sessions.

## Model files and troubleshooting

Keep `landslide_model4.pkl` and `landslide_scaler4.pkl` in the project root when launching the command above. The app uses relative paths, so launching it from another working directory can prevent the artifacts from loading.

If the model or scaler cannot be loaded, the interface remains available but predictions are disabled. Common fixes are:

- Confirm both `.pkl` files exist and were not renamed.
- Start Streamlit from the project root with `streamlit run src/app.py`.
- Use compatible versions of Python, scikit-learn, and joblib. Serialized scikit-learn artifacts can fail to load across incompatible versions.
- Only load model files from trusted sources; `joblib.load()` may execute arbitrary code embedded in an untrusted file.

## Development notes

- The repository currently contains the application and serialized artifacts, but no training pipeline or test suite.
- The dashboard’s recommendations are general guidance written in the UI, not calibrated operational instructions.
- To replace the model, train it with the same four named features in the exact order above and update the model and scaler together.
