
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

lgbm = joblib.load(BASE_DIR / "models" / "lightgbm.pkl")
xgb = joblib.load(BASE_DIR / "models" / "xgboost.pkl")


def ensemble_predict(X):

    lgb_pred = lgbm.predict(X)

    xgb_pred = xgb.predict(X)

    prediction = (
          0.55 * lgb_pred
        + 0.45 * xgb_pred
    )

    return prediction