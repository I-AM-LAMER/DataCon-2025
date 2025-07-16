import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/multitask_processed.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models/tuned")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False).replace([np.inf, -np.inf], np.nan)
X = df.drop(columns=[c for c in df if c.startswith(("ic50_","pIC50_"))]).select_dtypes(include=[np.number])
y = df["pIC50_bace1"].dropna()
X = X.loc[y.index]

param_dist = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
scorers = {
    "R2": make_scorer(r2_score),
    "RMSE": make_scorer(rmse, greater_is_better=False)
}

rs = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    scoring=scorers,
    refit="RMSE",
    cv=cv,
    verbose=2
)
rs.fit(X, y)

print("Best params:", rs.best_params_)
print("CV results:", rs.cv_results_["mean_test_RMSE"][rs.best_index_])

joblib.dump(rs.best_estimator_, os.path.join(MODELS_DIR, "rf_bace1_tuned.joblib"))
