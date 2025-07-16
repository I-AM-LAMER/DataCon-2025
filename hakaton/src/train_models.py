import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/multitask_processed.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.replace([np.inf, -np.inf], np.nan)

X_all = df.drop(
    columns=[c for c in df.columns if c.startswith("ic50_") or c.startswith("pIC50_")]
).select_dtypes(include=[np.number])

for target in ["bace1", "gsk3b", "ache"]:
    y_col = f"pIC50_{target}"
    mask = df[y_col].notna()
    X = X_all.loc[mask]
    y = df.loc[mask, y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rrmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{target}: R2={r2:.3f}, RMSE={rmse:.3f}")
    joblib.dump(model, os.path.join(MODELS_DIR, f"rf_{target}.joblib"))
