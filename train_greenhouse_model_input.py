import pandas as pd

# ✅ Load the model
import xgboost as xgb
model = xgb.XGBRegressor()
model.load_model("greenhouse_model.xgb")

# ✅ Feature names (same order used during training)
feature_names = [
    "MedInc",       # Median income
    "HouseAge",     # Median house age
    "AveRooms",     # Avg rooms per household
    "AveBedrms",    # Avg bedrooms per household
    "Population",   # Total population
    "AveOccup",     # Avg occupants per household
    "Latitude",     # Latitude
    "Longitude"     # Longitude
]

# ✅ Example input values (you can later replace these with dynamic input)
inputs = [3.5, 25, 6.1, 1.1, 980.0, 2.6, 37.77, -122.42]  # <-- REPLACE with your own values or input()

# ✅ Convert to DataFrame
input_df = pd.DataFrame([inputs], columns=feature_names)

# ✅ Predict
prediction = model.predict(input_df)[0]
print(f"✅ Predicted House Value: ${prediction * 100000:.2f}")






