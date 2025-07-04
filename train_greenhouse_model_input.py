import streamlit as st
import pandas as pd
import xgboost as xgb

# Load the model (ensure train_greenhouse_model.json is in same directory)
model = xgb.XGBRegressor()
model.load_model("train_greenhouse_model.json")

# 🌿 Page setup
st.set_page_config(page_title="GHG Emission Predictor", layout="centered")
st.title("🌱 Greenhouse Gas Emission Predictor")

# 🚀 Input form
with st.form("emission_form"):
    st.subheader("Enter Input Details")

    col1, col2 = st.columns(2)

    with col1:
        industry = st.selectbox("Select Industry", ["Electricity", "Cement", "Fertilizer", "Textile", "Transportation"])
        substance = st.selectbox("Substance", ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6"])
        margin = st.number_input("Margin of Supply Chain Emission Factor", min_value=0.0, format="%.4f")

    with col2:
        dq_reliability = st.slider("DQ Reliability Score", 0.0, 1.0, 0.8)
        dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0, 0.8)
        dq_geographical = st.slider("DQ Geographical Correlation", 0.0, 1.0, 0.8)
        dq_technological = st.slider("DQ Technological Correlation", 0.0, 1.0, 0.8)
        dq_datacollection = st.slider("DQ Data Collection", 0.0, 1.0, 0.8)

    # ✅ Submit button
    submitted = st.form_submit_button("🔍 Predict Emissions")

# When submitted
if submitted:
    with st.spinner("Predicting emissions..."):
        # Prepare input
        input_data = pd.DataFrame([{
            "industry_code": hash(industry) % 10000,
            "substance": hash(substance) % 10000,
            "margins_of_supply_chain_emission_factors": margin,
            "dq_reliabilityscore_of_factors_without_margins": dq_reliability,
            "dq_temporalcorrelation_of_factors_without_margins": dq_temporal,
            "dq_geographicalcorrelation_of_factors_without_margins": dq_geographical,
            "dq_technologicalcorrelation_of_factors_without_margins": dq_technological,
            "dq_datacollection_of_factors_without_margins": dq_datacollection
        }])

        # Prediction
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted GHG Emission Factor: **{prediction:.4f} kg CO₂-eq/unit**")



