#!/usr/bin/env python
# blend_prediction.py
# Shell.ai Hackathon – Fuel‑Blend Properties Prediction
# Author: Safwan (stock‑gpt) – July 2025

import os, hashlib, warnings, argparse
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna, joblib, json, random

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------------------------------
# 1. CLI args
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train", default="train.csv")
parser.add_argument("--test",  default="test.csv")
parser.add_argument("--out",   default="submission.csv")
parser.add_argument("--trials", type=int, default=60, help="Optuna trials per target")
parser.add_argument("--lgb_weight", type=float, default=0.6, help="Blend weight for LightGBM")
args = parser.parse_args()

# ------------------------------------------------------------------------------
# 2. Load
# ------------------------------------------------------------------------------
train = pd.read_csv(args.train)
test  = pd.read_csv(args.test)

X_base_train = train.iloc[:, :55].copy()
y_train      = train.iloc[:, 55:].copy()
X_base_test  = test.iloc[:, :55].copy()

# ------------------------------------------------------------------------------
# 3. Feature engineering
# ------------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # weighted component‑property
    for comp in range(1, 6):
        f_col = f"Component{comp}_fraction"
        if f_col not in df: continue
        for prop in range(1, 11):
            p_col = f"Component{comp}_Property{prop}"
            new_c = f"W_Comp{comp}_Prop{prop}"
            out[new_c] = df[f_col] * df.get(p_col, 0)
    # pairwise fraction interactions
    for i in range(1, 6):
        for j in range(i+1, 6):
            fi, fj = f"Component{i}_fraction", f"Component{j}_fraction"
            out[f"Frac_{i}_{j}"] = df.get(fi, 0) * df.get(fj, 0)
    return out

X_train = add_features(X_base_train)
X_test  = add_features(X_base_test)

# ensure same columns order
missing = set(X_train.columns) - set(X_test.columns)
for c in missing: X_test[c] = 0
X_test = X_test[X_train.columns]

# ------------------------------------------------------------------------------
# 4. Scaling
# ------------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

# ------------------------------------------------------------------------------
# 5. GroupKFold groups (hash of fractions)
# ------------------------------------------------------------------------------
def hash_frac(row, prec=2):
    key = tuple(np.round(row[[f"Component{i}_fraction" for i in range(1,6)]], prec))
    return int(hashlib.md5(str(key).encode()).hexdigest(),16)%10_000_000

groups = X_base_train.apply(hash_frac, axis=1)
gkf = GroupKFold(n_splits=5)

# ------------------------------------------------------------------------------
# 6. LightGBM + Optuna tuning (bag 5 folds)
# ------------------------------------------------------------------------------
device = "gpu" if lgb.get_device_name(0) else "cpu"
print(f"🔧 Using LightGBM on {device.upper()}")

lgb_models = {t: [] for t in y_train.columns}
val_mapes  = {}
study_db   = optuna.storages.InMemoryStorage()

def objective_factory(X, y):
    def obj(trial):
        params = {
            "objective": "rmse",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "device_type": device,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("leaves", 31, 1023, log=True),
            "feature_fraction": trial.suggest_float("feat_frac", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bag_frac", 0.5, 1.0),
            "bagging_freq": 1,
            "min_data_in_leaf": trial.suggest_int("min_leaf", 20, 200),
            "lambda_l1": trial.suggest_float("l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("l2", 0.0, 5.0),
            "seed": SEED,
        }
        mape_scores=[]
        for tr, vl in gkf.split(X, y, groups):
            m = lgb.train(params, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
                          num_boost_round=2000,
                          valid_sets=[lgb.Dataset(X.iloc[vl], y.iloc[vl])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p = m.predict(X.iloc[vl])
            mape_scores.append(mean_absolute_percentage_error(y.iloc[vl], p))
        return np.mean(mape_scores)
    return obj

for tgt in y_train.columns:
    print(f"\n🔎 Tuning LightGBM for {tgt} …")
    study = optuna.create_study(direction="minimize", storage=study_db, sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_factory(X_train, y_train[tgt]), n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial.params
    best.update({"objective":"rmse","metric":"mae","verbosity":-1,"device_type":device,"seed":SEED})
    fold_mapes=[]
    for fold,(tr,vl) in enumerate(gkf.split(X_train, y_train[tgt], groups)):
        mdl = lgb.train(best, lgb.Dataset(X_train.iloc[tr], y_train[tgt].iloc[tr]),
                        num_boost_round=study.best_trial.user_attrs.get("best_iteration", 1000))
        lgb_models[tgt].append(mdl)
        pr = mdl.predict(X_train.iloc[vl])
        fold_mapes.append(mean_absolute_percentage_error(y_train[tgt].iloc[vl], pr))
    val_mapes[tgt] = np.mean(fold_mapes)
    print(f"📊 {tgt} CV‑MAPE {val_mapes[tgt]:.4f}")

print(f"\n📈 Mean CV‑MAPE: {np.mean(list(val_mapes.values())):.4f}")

# ------------------------------------------------------------------------------
# 7. CatBoost quick model (no tuning, GPU if available)
# ------------------------------------------------------------------------------
cat_models = {}
has_gpu = CatBoostRegressor().get_param("task_type")=="GPU" if os.getenv("CUDA_VISIBLE_DEVICES") else False
cat_params = dict(
    iterations=1200,
    depth=8,
    learning_rate=0.05,
    loss_function="MAE",
    task_type="GPU" if has_gpu else "CPU",
    verbose=False,
    random_seed=SEED,
)

print(f"\n🚂 Training CatBoost ({'GPU' if has_gpu else 'CPU'}) …")
for tgt in y_train.columns:
    cat = CatBoostRegressor(**cat_params)
    cat.fit(X_train, y_train[tgt])
    cat_models[tgt] = cat

# ------------------------------------------------------------------------------
# 8. Predict & blend
# ------------------------------------------------------------------------------
preds_lgb = {}
for tgt, mdl_list in lgb_models.items():
    fold_preds = np.mean([m.predict(X_test) for m in mdl_list], axis=0)
    preds_lgb[tgt] = fold_preds

preds_cat = {tgt: mdl.predict(X_test) for tgt, mdl in cat_models.items()}

alpha = args.lgb_weight
blend_preds = {tgt: alpha*preds_lgb[tgt] + (1-alpha)*preds_cat[tgt] for tgt in y_train.columns}

submission = pd.DataFrame(blend_preds)
submission.insert(0, "ID", test["ID"])
submission.to_csv(args.out, index=False)

print(f"\n✅ Submission saved to “{args.out}”")
print("   You can now upload it to the leaderboard.\n")
