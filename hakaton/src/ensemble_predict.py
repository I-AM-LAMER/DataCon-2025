import os
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor

BASE = os.path.dirname(__file__)
rf_path = os.path.join(BASE, "../models/rf_bace1.joblib")
xgb_path = os.path.join(BASE, "../models/xgb_bace1.joblib")
df = pd.read_csv(os.path.join(BASE, "../data/multitask_processed.csv"), low_memory=False)

rf = joblib.load(rf_path)
xgb = joblib.load(xgb_path)

X = df.drop(columns=[c for c in df if c.startswith(("ic50_","pIC50_"))]).select_dtypes(include=[np.number])

pred_rf = rf.predict(X)
pred_xgb = xgb.predict(X)
pred_ens = (pred_rf + pred_xgb) / 2

df["pred_RF"] = pred_rf
df["pred_XGB"] = pred_xgb
df["pred_ensemble"] = pred_ens

df.to_csv(os.path.join(BASE, "../results/ensemble_predictions.csv"), index=False)
print("Saved ensemble predictions")
