
import xgboost as xgb
import streamlit as st
import pickle
import pandas as pd

# Load model safely
try:
    with open("train_greenhouse_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure train_greenhouse_model.pkl is uploaded.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load the model
model = xgb.XGBRegressor()
# Load the model using pickle
with open("train_greenhouse_model.pkl", "rb") as file:
    model = pickle.load(file)

# Set up the page
st.set_page_config(page_title="GHG Emission Predictor", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🌿 Greenhouse Gas Emission Predictor</h1>
    <hr>
""", unsafe_allow_html=True)

# Arrange input nicely
with st.form("ghg_form"):
    col1, col2 = st.columns(2)

    with col1:
        industry = st.selectbox("🏭 Select Industry", ["Electricity", "Cement", "Fertilizer", "Textile", "Transportation"])
        substance = st.selectbox("🧪 Substance", ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6"])
        margin = st.number_input("📉 Margin of Supply Chain Emission Factor", min_value=0.0, format="%.4f")

    with col2:
        dq_reliability = st.slider("🔍 DQ Reliability Score", 0.0, 1.0, 0.8)
        dq_temporal = st.slider("🕒 DQ Temporal Correlation", 0.0, 1.0, 0.8)
        dq_geographical = st.slider("🗺️ DQ Geographical Correlation", 0.0, 1.0, 0.8)
        dq_technological = st.slider("💻 DQ Technological Correlation", 0.0, 1.0, 0.8)
        dq_datacollection = st.slider("📦 DQ Data Collection", 0.0, 1.0, 0.8)

    submitted = st.form_submit_button("📊 Predict Emissions")

# When Submit button is pressed
if submitted:
    with st.spinner("Predicting emission factor..."):
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

        prediction = model.predict(input_data)[0]

    st.success(f"✅ Predicted Emission Factor: **{prediction:.4f} kg CO₂-eq/unit**")



