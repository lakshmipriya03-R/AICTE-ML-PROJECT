import pandas as pd

# ✅ Step 1: Define features in correct order
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

# ✅ Step 2: Collect your user inputs into a list (modify this as per your UI)
inputs = [float(val1), float(val2), float(val3), float(val4), float(val5), float(val6), float(val7), float(val8)]

# ✅ Step 3: Convert to a DataFrame with matching feature names
input_df = pd.DataFrame([inputs], columns=feature_names)

# ✅ Step 4: Make prediction
prediction = model.predict(input_df)[0]
print(f"Predicted house value: ${prediction * 100000:.2f}")






