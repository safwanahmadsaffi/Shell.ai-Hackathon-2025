import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np

#Load datasets
train_df = pd.read_csv("D:\dataset\train.csv")
test_df = pd.read_csv("D:\dataset\test.csv")

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Train columns: {list(train_df.columns)}")
print(f"Test columns: {list(test_df.columns)}")

# Separate 
X_train = train_df.drop([f"BlendProperty{i}" for i in range(1, 11)], axis=1)
y_train = train_df[[f"BlendProperty{i}" for i in range(1, 11)]]

#test
X_test = test_df.drop("ID", axis=1)

# use KNN 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using k=5
model = KNeighborsRegressor(n_neighbors=5, weights='distance')

# train and target
predictions = np.zeros((X_test.shape[0], y_train.shape[1]))

for i, col in enumerate(y_train.columns):
    print(f"Training KNN model for {col}")
    model.fit(X_train_scaled, y_train[col])
    predictions[:, i] = model.predict(X_test_scaled)

# Create a submit 
submission_df = pd.DataFrame(predictions, columns=[f"BlendProperty{i}" for i in range(1, 11)])

#save
submission_df.to_csv("submission.csv", index=False)
print("Optimized model trained and predictions saved to submission.csv")

