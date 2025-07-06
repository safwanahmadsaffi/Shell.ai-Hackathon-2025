import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Feature engineering function
def add_extra_features(df):
    # identify numeric features excluding ID and target columns
    features = [col for col in df.columns if col not in ['ID'] and not col.startswith('BlendProperty')]
    df['feature_sum'] = df[features].sum(axis=1)
    df['feature_mean'] = df[features].mean(axis=1)
    df['feature_std'] = df[features].std(axis=1)
    return df

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("/content/test.csv")

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Test columns: {list(test_df.columns)}")

# Separate features (X) and target variables (y)
X_train = train_df.drop([f"BlendProperty{i}" for i in range(1, 11)], axis=1)
y_train = train_df[[f"BlendProperty{i}" for i in range(1, 11)]]

# For the test set, drop the 'ID' column
X_test = test_df.drop("ID", axis=1)

# Add extra statistical features to improve model
X_train = add_extra_features(X_train)
X_test = add_extra_features(X_test)

# Initialize models for ensemble
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
knn_model = KNeighborsRegressor(n_neighbors=7)

# Prepare container for predictions
predictions = np.zeros((X_test.shape[0], y_train.shape[1]))

# Cross-validation setup
gkf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, col in enumerate(y_train.columns):
    print(f"Training and validating model for {col}")
    cv_scores = []
    for train_idx, val_idx in gkf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[col].iloc[train_idx], y_train[col].iloc[val_idx]
        gb_model.fit(X_tr, y_tr)
        preds = gb_model.predict(X_val)
        cv_scores.append(mean_absolute_percentage_error(y_val, preds))
    print(f"{col} CV MAPE: {np.mean(cv_scores):.5f}")
    # Train on full data
    gb_model.fit(X_train, y_train[col])
    knn_model.fit(X_train, y_train[col])
    preds_gb = gb_model.predict(X_test)
    preds_knn = knn_model.predict(X_test)
    predictions[:, i] = (preds_gb + preds_knn) / 2

# Create and save submission
submission_df = pd.DataFrame(predictions, columns=[f"BlendProperty{i}" for i in range(1, 11)])
submission_df.to_csv("submission.csv", index=False)
print("Ensembled model trained and predictions saved to submission.csv")

