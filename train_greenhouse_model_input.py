import streamlit as st
import pandas as pd
import xgboost as xgb

# ✅ Load the trained model
model = xgb.XGBRegressor()
model.load_model("greenhouse_model.xgb")

# ✅ Feature names (must match training order)
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

# ✅ Page title
st.title("🏡 Greenhouse Model - Real-Time House Value Predictor")
st.markdown("🔢 Enter California housing features to predict house value (trained with XGBoost)")

# ✅ Input fields
inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}:", value=1.0, step=0.1)
    inputs.append(val)

# ✅ Prediction on button click
if st.button("Predict House Value"):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted House Value: **${prediction * 100000:.2f}**")







