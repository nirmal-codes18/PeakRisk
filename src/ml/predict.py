import pandas as pd
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "peakrisk_model.joblib"
)

model = joblib.load(MODEL_PATH)


def predict_expedition_success(
    year,
    season,
    heightm,
    region,
    open_status,
    trekking,
    total_members,
    camps
):
    input_data = pd.DataFrame([{
        "YEAR": year,
        "SEASON": season,
        "HEIGHTM": heightm,
        "REGION": region,
        "OPEN": open_status,
        "TREKKING": trekking,
        "TOTMEMBERS": total_members,
        "CAMPS": camps
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    return {
        "prediction": bool(prediction),
        "success_probability": float(probability[1]),
        "failure_probability": float(probability[0])
    }