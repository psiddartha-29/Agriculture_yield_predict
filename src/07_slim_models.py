"""
Train SLIM versions of the three models so the API fits in Render's 512 MiB RAM.

Why: the original tuned models (min_samples_leaf=1-2, 80-100 trees) used ~520 MB
of RAM once loaded. Larger leaves + fewer trees cut that ~8x with a small
accuracy change.

Run from project root:  python src/07_slim_models.py
Trains on years <= 2018, reports metrics on years >= 2019, then refits on ALL
data and saves the final models into models/ (same filenames the API loads).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA = BASE_DIR / "data" / "processed"
MODELS = BASE_DIR / "models"
MODELS.mkdir(exist_ok=True)
HECTARE_TO_ACRE = 2.47105
CATS = ["Crop", "Season", "State"]


def ohe():
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def yield_data():
    df = pd.read_csv(DATA / "crop_yield_processed.csv").dropna(subset=["Yield_kg_per_acre", "Crop_Year"])
    df = df.sort_values("Crop_Year").reset_index(drop=True)
    df["Fertilizer_per_acre"] = df["Fertilizer_kg"] / df["Area_acres"].replace(0, np.nan)
    df["Pesticide_per_acre"] = df["Pesticide_kg"] / df["Area_acres"].replace(0, np.nan)
    feats = ["Crop", "Crop_Year", "Season", "State", "Area_acres", "Annual_Rainfall_mm",
             "Fertilizer_kg", "Pesticide_kg", "Fertilizer_per_acre", "Pesticide_per_acre"]
    nums = [c for c in feats if c not in CATS]
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), nums),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", ohe())]), CATS),
    ])
    model = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=5, max_features=0.7, n_jobs=1, random_state=42)
    return df, feats, "Yield_kg_per_acre", pre, model, "best_yield_prediction_model.pkl"


def fert_data():
    df = pd.read_csv(DATA / "Fertilizer.csv")
    df["Area_acres"] = df["Area"] * HECTARE_TO_ACRE
    df = df.sort_values("Crop_Year").reset_index(drop=True)
    feats = ["Crop", "Crop_Year", "Season", "State", "Area_acres", "Annual_Rainfall"]
    pre = ColumnTransformer([
        ("categorical", ohe(), CATS),
        ("numerical", StandardScaler(), ["Crop_Year", "Area_acres", "Annual_Rainfall"]),
    ])
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, max_features=1.0, n_jobs=1, random_state=42)
    return df, feats, "Fertilizer", pre, model, "fertilizer_prediction_model1.pkl"


def pest_data():
    df = pd.read_csv(DATA / "Pesticide.csv")
    df["Area_acres"] = df["Area"] * HECTARE_TO_ACRE
    df["Annual_Rainfall_mm"] = df["Annual_Rainfall"]
    df = df.sort_values("Crop_Year").reset_index(drop=True)
    feats = ["Crop", "Crop_Year", "Season", "State", "Area_acres", "Annual_Rainfall_mm"]
    pre = ColumnTransformer([
        ("categorical", ohe(), CATS),
        ("numerical", "passthrough", ["Crop_Year", "Area_acres", "Annual_Rainfall_mm"]),
    ])
    # pesticide needs small leaves to stay accurate, so use fewer trees instead
    model = ExtraTreesRegressor(n_estimators=30, min_samples_leaf=2, n_jobs=1, random_state=42)
    return df, feats, "Pesticide", pre, model, "pesticide_prediction_model.pkl"


for name, loader in [("yield", yield_data), ("fertilizer", fert_data), ("pesticide", pest_data)]:
    df, feats, target, pre, model, filename = loader()
    pipe = Pipeline([("preprocessor", pre), ("model", model)])

    train = df[df["Crop_Year"] <= 2018]
    test = df[df["Crop_Year"] >= 2019]
    pipe.fit(train[feats], train[target])
    pred = pipe.predict(test[feats])
    print(f"{name:10s} test MAE={mean_absolute_error(test[target], pred):,.2f}  R2={r2_score(test[target], pred):.4f}")

    pipe.fit(df[feats], df[target])          # final refit on all data
    joblib.dump(pipe, MODELS / filename, compress=3)   # compress=3 also shrinks the file
    print(f"           saved {filename}: {(MODELS / filename).stat().st_size / 1e6:.1f} MB")
