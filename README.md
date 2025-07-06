# 🚀 Shell.ai Hackathon 2025 - Fuel Blend Properties Prediction

## 🎯 Challenge Overview
Predict **10 fuel blend properties** from **55 input features** to enable sustainable aviation fuel optimization.

## 📊 Dataset
- **Train**: 65 columns (55 features + 10 targets)
- **Test**: 500 samples, 55 features only
- **Features**: 5 blend compositions + 50 component properties
- **Targets**: BlendProperty1 through BlendProperty10

## 📈 Evaluation
- **Metric**: MAPE (Mean Absolute Percentage Error)
- **Scoring**: `Score = max(0, 100 - (MAPE / reference_cost) × 100)`
- **Reference Costs**: Public: 2.72 | Private: 2.58

## 📝 Submission Requirements
- **Format**: CSV with 500 rows × 10 columns
- **Columns**: BlendProperty1-10 (exact names)
- **Data**: Floating-point numbers only
- **No ID column** or additional columns

## ⚡ Key Challenges
- Complex fuel component interactions
- High-dimensional feature space (55 features)
- Precise predictions needed for safety/performance

## 🌍 Impact
- Accelerate sustainable fuel adoption
- Reduce aviation environmental footprint
- Enable real-time blend optimization

## 🛠️ Quick Start
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Split features and targets
X_train = train.iloc[:, :55]
y_train = train.iloc[:, 55:]

# Train and predict
model = RandomForestRegressor()
model.fit(X_train, y_train)
predictions = model.predict(test)

# Create submission
submission = pd.DataFrame(predictions, 
                         columns=[f'BlendProperty{i}' for i in range(1, 11)])
submission.to_csv('submission.csv', index=False)
```

## 📧 Contact
ai@shell.com

---
*Building a sustainable future through AI* 🌱