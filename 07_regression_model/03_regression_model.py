# Mancy Chen 20-03-2026
# XGBoost in GPU

import numpy as np
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import KFold, cross_validate

# -----------------------------
# Tiny test dataset: 20 samples, 5 features
# y ranges from 1 to 30
# -----------------------------
X = [
    [0.1, 1.2, 2.1, 0.5, 3.0],
    [0.2, 1.0, 2.0, 0.4, 2.8],
    [0.3, 0.9, 1.8, 0.6, 2.7],
    [0.4, 1.1, 2.2, 0.7, 3.1],
    [0.5, 1.3, 2.4, 0.8, 3.3],
    [0.6, 1.5, 2.5, 0.9, 3.5],
    [0.7, 1.4, 2.6, 1.0, 3.7],
    [0.8, 1.6, 2.8, 1.1, 3.8],
    [0.9, 1.7, 3.0, 1.2, 4.0],
    [1.0, 1.8, 3.1, 1.3, 4.2],
    [1.1, 1.9, 3.3, 1.4, 4.4],
    [1.2, 2.0, 3.5, 1.5, 4.5],
    [1.3, 2.1, 3.6, 1.6, 4.7],
    [1.4, 2.2, 3.8, 1.7, 4.9],
    [1.5, 2.3, 4.0, 1.8, 5.0],
    [1.6, 2.4, 4.1, 1.9, 5.2],
    [1.7, 2.5, 4.3, 2.0, 5.4],
    [1.8, 2.6, 4.5, 2.1, 5.6],
    [1.9, 2.7, 4.6, 2.2, 5.7],
    [2.0, 2.8, 4.8, 2.3, 5.9],
]

y = [1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 18, 20, 21, 23, 24, 26, 28, 30]

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# -----------------------------
# Base XGBoost regressor on GPU
# -----------------------------
xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    device="cuda",          # GPU
    eval_metric="rmse",
    random_state=42
)

# -----------------------------
# Search space for BayesSearchCV
# Keep it small for a quick test
# -----------------------------
search_spaces = {
    "n_estimators": Integer(30, 120),
    "max_depth": Integer(2, 6),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "subsample": Real(0.6, 1.0),
    "colsample_bytree": Real(0.6, 1.0),
    "min_child_weight": Integer(1, 8),
    "reg_alpha": Real(1e-8, 1.0, prior="log-uniform"),
    "reg_lambda": Real(1e-6, 10.0, prior="log-uniform"),
}

# -----------------------------
# Inner CV: 5-fold
# Outer CV: 3-fold
# -----------------------------
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Important:
# n_jobs=1 is usually best for a single GPU
bayes_search = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_spaces,
    n_iter=8,                   # small for quick testing
    scoring="neg_root_mean_squared_error",
    cv=inner_cv,
    n_jobs=1,
    random_state=42,
    verbose=1,
    refit=True
)

# -----------------------------
# Nested CV
# -----------------------------
nested_scores = cross_validate(
    bayes_search,
    X,
    y,
    cv=outer_cv,
    scoring=("neg_root_mean_squared_error", "r2"),
    return_estimator=True,
    n_jobs=1
)

# -----------------------------
# Results
# -----------------------------
rmse_scores = -nested_scores["test_neg_root_mean_squared_error"]
r2_scores = nested_scores["test_r2"]

print("Outer RMSE per fold:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

print("Outer R2 per fold:", r2_scores)
print("Mean R2:", r2_scores.mean())

# Best params from each outer fold
for i, est in enumerate(nested_scores["estimator"], start=1):
    print(f"\nOuter fold {i} best params:")
    print(est.best_params_)

# Optional quick GPU smoke test on full data
best_model = bayes_search.fit(X, y).best_estimator_
print("\nFinal best model fitted on full dataset:")
print(best_model)