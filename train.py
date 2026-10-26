"""Train the food delivery time model and save it as one reusable pipeline.

Run:  python train.py
Creates: model/food_delivery_model.joblib and model/metrics.json

This reproduces the notebook's cleaning and feature steps, with two fixes:
the train/test split happens BEFORE imputation (no test-set leakage), and
the text "NaN" in the weather column is treated as a missing value.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_PATH = Path("Food delivery.csv")
MODEL_DIR = Path("model")

TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
NUMERIC = [
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Vehicle_condition", "multiple_deliveries", "distance_km",
]
ORDINAL = ["Road_traffic_density"]
CATEGORICAL = ["Type_of_order", "Type_of_vehicle", "Festival", "City", "Weatherconditions"]
FEATURES = NUMERIC + ORDINAL + CATEGORICAL
TARGET = "Time_taken(min)"


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two coordinate pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def clean_raw(df):
    """Fix the messy text in the raw CSV. Missing values are left for the pipeline."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("NaN", np.nan)
    # "conditions Sunny" -> "Sunny"
    df["Weatherconditions"] = (
        df["Weatherconditions"].str.replace("conditions ", "", regex=False).replace("NaN", np.nan)
    )
    # "(min) 24" -> 24
    df[TARGET] = pd.to_numeric(df[TARGET].str.replace("(min) ", "", regex=False), errors="coerce")
    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].clip(upper=5)
    df["Road_traffic_density"] = df["Road_traffic_density"].map(TRAFFIC_MAP)
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )
    df = df[df["distance_km"] < 100]          # drop impossible GPS rows
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def build_pipeline(model):
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("ord", SimpleImputer(strategy="most_frequent"), ORDINAL),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([("prep", preprocess), ("model", model)])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    return {"r2": float(r2_score(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred)))}


def main():
    df = clean_raw(pd.read_csv(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

    # Hyperparameters found by RandomizedSearchCV in the notebook.
    candidates = {
        "Random Forest (tuned)": RandomForestRegressor(
            n_estimators=50, max_depth=15, min_samples_split=5, min_samples_leaf=4,
            max_features=0.5, random_state=42, n_jobs=-1),
        "XGBoost (tuned)": XGBRegressor(
            n_estimators=150, max_depth=7, learning_rate=0.03, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=1, reg_lambda=2, random_state=42, n_jobs=-1),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        results[name] = {"train": evaluate(pipe, X_train, y_train), "test": evaluate(pipe, X_test, y_test)}
        fitted[name] = pipe
        print(f"{name}: test R2={results[name]['test']['r2']:.3f}, "
              f"test RMSE={results[name]['test']['rmse']:.2f} min, train R2={results[name]['train']['r2']:.3f}")

    best = max(results, key=lambda n: results[n]["test"]["r2"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], MODEL_DIR / "food_delivery_model.joblib", compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps({"best_model": best, "results": results}, indent=2))
    print(f"Saved best model: {best}")


if __name__ == "__main__":
    main()
