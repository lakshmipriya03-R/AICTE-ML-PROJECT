import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the model
model = xgb.XGBRegressor()
model.load_model("greenhouse_model.xgb")
  # Make sure this file is in the same directory

# Streamlit UI
st.set_page_config(page_title="GHG Emission Predictor", layout="centered")
st.title("🌱 Greenhouse Gas Emission Predictor")

st.markdown("Enter the details below to predict emission factor:")

industry = st.selectbox("Select Industry", ["Electricity", "Cement", "Transport", "Fertilizer"])
ghg = st.selectbox("Select Gas", ["CO2", "CH4", "N2O"])
margin = st.slider("Enter Margin", 0.0, 10.0, 1.0)
dq_assessment = st.slider("DQ Assessment Score", 0.0, 1.0, 0.5)
dq_coverage = st.slider("DQ Coverage Score", 0.0, 1.0, 0.5)
dq_uncertainty = st.slider("DQ Uncertainty Score", 0.0, 1.0, 0.5)
dq_verification = st.slider("DQ Verification Score", 0.0, 1.0, 0.5)

# Predict on button click
if st.button("🚀 Predict Emission"):
    input_df = pd.DataFrame({
        'industry': [industry],
        'ghg': [ghg],
        'margin': [margin],
        'dq_assessment': [dq_assessment],
        'dq_coverage': [dq_coverage],
        'dq_uncertainty': [dq_uncertainty],
        'dq_verification': [dq_verification]
    })

    # Convert categorical to dummy
    input_df = pd.get_dummies(input_df)
    
    # Align with model input features
    model_features = [
        'margin', 'dq_assessment', 'dq_coverage',
        'dq_uncertainty', 'dq_verification',
        'industry_Cement', 'industry_Electricity', 'industry_Fertilizer', 'industry_Transport',
        'ghg_CH4', 'ghg_CO2', 'ghg_N2O'
    ]
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    prediction = model.predict(input_df)[0]
    st.success(f"🌍 Predicted Emission Factor: **{prediction:.4f}**")





