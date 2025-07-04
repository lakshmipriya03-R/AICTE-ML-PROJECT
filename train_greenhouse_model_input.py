import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle  # or use open() with rb if you're avoiding joblib

# Load the model
model = xgb.XGBRegressor()
model.load_model("train_greenhouse_model.json")  # or .pkl if you're using pickle

# UI
def main():
    st.set_page_config(page_title="GHG Emission Predictor", layout="centered")
    st.title("🌱 Greenhouse Gas Emission Predictor")
    
    st.markdown("### Enter details to predict GHG Emission Factor")

    # Sample inputs (you can add more features here as per your model)
    industry = st.selectbox("Industry", ["Electricity", "Cement", "Steel", "Fertilizer"])
    dq_score = st.slider("Data Quality Score", 0.0, 1.0, 0.5)
    margin = st.number_input("Margin of Supply Chain Emission Factors", min_value=0.0, max_value=1.0)

    # Combine into input for model
    input_data = pd.DataFrame([[dq_score, margin]], columns=["dq_score", "margins"])
    
    if st.button("Predict"):
        prediction = model.predict(input_data)
        st.success(f"🌍 Predicted Emission Factor: **{prediction[0]:.4f}**")

# ✅ Add this line at the end of the file:
if __name__ == "__main__":
    main()

