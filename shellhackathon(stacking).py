#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#  Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.iloc[:, :55].copy()
y = train.iloc[:, 55:]
test_X = test.iloc[:, :55].copy()
if 'ID' in test.columns:
    test_X = test.drop(columns=['ID']).copy()

# Features
for prop_num in [2, 3]:
    cols = [f'Component{i}_Property{prop_num}' for i in range(1,6)]
    X[f'Avg_Property{prop_num}'] = X[cols].mean(axis=1)
    test_X[f'Avg_Property{prop_num}'] = test_X[cols].mean(axis=1)

for prop_num in range(1, 11):
    comp_props = [f'Component{i}_Property{prop_num}' for i in range(1,6)]
    comp_fracs = [f'Component{i}_fraction' for i in range(1,6)]
    X[f'WeightedAvg_Property{prop_num}'] = sum(
        X[frac] * X[prop] for frac, prop in zip(comp_fracs, comp_props))
    test_X[f'WeightedAvg_Property{prop_num}'] = sum(
        test_X[frac] * test_X[prop] for frac, prop in zip(comp_fracs, comp_props))

#  KFold for stacking
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to collect out-of-fold predictions (for train) and test predictions
meta_train = np.zeros((X.shape[0], 10*5))  # 5 base models × 10 targets
meta_test = np.zeros((test_X.shape[0], 10*5))

# Base models
base_models = [
    ('xgb', MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='mae',
        random_state=42, verbosity=0))),
    ('lgb', MultiOutputRegressor(LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1))),
    ('rf', MultiOutputRegressor(RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42))),
    ('ridge', MultiOutputRegressor(Ridge())),
    ('cat', MultiOutputRegressor(CatBoostRegressor(
        verbose=0, iterations=500, depth=6, learning_rate=0.05, random_seed=42)))
]

# Out-of-fold predictions
for idx, (name, model) in enumerate(base_models):
    print(f"✅ Training base model: {name}")
    test_preds_fold = np.zeros((test_X.shape[0], 10, 5))  # test preds per fold

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        meta_train[valid_idx, idx*10:(idx+1)*10] = model.predict(X_valid)
        test_preds_fold[:, :, fold] = model.predict(test_X)

    # Average test preds across folds
    meta_test[:, idx*10:(idx+1)*10] = test_preds_fold.mean(axis=2)

print(" Meta features shape:", meta_train.shape, meta_test.shape)

#  Meta-model: simple Ridge (works well)
meta_model = MultiOutputRegressor(Ridge())
meta_model.fit(meta_train, y)
meta_train_preds = meta_model.predict(meta_train)
meta_test_preds = meta_model.predict(meta_test)

#  Evaluate
stacking_mape = mean_absolute_percentage_error(y, meta_train_preds)
print("✅ Stacking train MAPE:", stacking_mape)

# Save submission
submission = pd.DataFrame(
    meta_test_preds,
    columns=[f"BlendProperty{i}" for i in range(1, 11)]
)
if 'ID' in test.columns:
    submission.insert(0, 'ID', test['ID'])
submission.to_csv("submission_stacking.csv", index=False)
print(" submission_stacking.csv saved! Shape:", submission.shape)

