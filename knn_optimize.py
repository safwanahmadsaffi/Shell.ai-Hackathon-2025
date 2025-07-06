import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Test columns: {list(test_df.columns)}")

# Separate features (X) and target variables (y)
X_train = train_df.drop([f"BlendProperty{i}" for i in range(1, 11)], axis=1)
y_train = train_df[[f"BlendProperty{i}" for i in range(1, 11)]]

# For the test set, we need to drop the 'ID' column as it's not a feature for the model
X_test = test_df.drop("ID", axis=1)

# Scale the features for KNN (important for distance-based algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the K-Nearest Neighbors Regressor model
# Using k=5 as a starting point, can be tuned further for better performance
model = KNeighborsRegressor(n_neighbors=5, weights='distance')

# Since KNeighborsRegressor can handle multi-output, we can train one model for all targets
# But for consistency and potential optimization per target, we'll train one model per target property
predictions = np.zeros((X_test.shape[0], y_train.shape[1]))

for i, col in enumerate(y_train.columns):
    print(f"Training KNN model for {col}")
    model.fit(X_train_scaled, y_train[col])
    predictions[:, i] = model.predict(X_test_scaled)

# Create a submission DataFrame
submission_df = pd.DataFrame(predictions, columns=[f"BlendProperty{i}" for i in range(1, 11)])

# Save the submission file
submission_df.to_csv("submission.csv", index=False)
print("Optimized model trained and predictions saved to submission.csv")

