import pandas as pd
from xgboost import XGBRegressor
import pickle

# ✅ Create the training dataset directly
data = pd.DataFrame({
    "industry_code": [101, 102, 103, 104],
    "substance": [1, 2, 1, 3],
    "dq_reliability": [0.9, 0.8, 0.85, 0.95],
    "dq_temporal": [0.7, 0.6, 0.65, 0.75],
    "dq_geographical": [0.8, 0.85, 0.75, 0.9],
    "dq_technological": [0.85, 0.8, 0.9, 0.95],
    "dq_datacollection": [0.95, 0.9, 0.85, 0.98],
    "margin": [0.01, 0.02, 0.015, 0.018],
    "emission_with_margins": [0.25, 0.3, 0.28, 0.27]
})

# ✅ Features and Target
X = data.drop("emission_with_margins", axis=1)
y = data["emission_with_margins"]

# ✅ Train Model
model = XGBRegressor()
model.fit(X, y)

# ✅ Save model
with open("greenhouse_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as greenhouse_model.pkl")
