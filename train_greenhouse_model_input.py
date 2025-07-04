import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("train_greenhouse_model.json")


# 🔹 Streamlit UI
def main():
    st.set_page_config(page_title="GHG Emission Predictor", layout="centered")
    st.title("🌱 Greenhouse Gas Emission Predictor")

    st.markdown("Enter relevant details below to predict **Supply Chain Emission Factor with Margins**")

    # 🧾 Inputs from user
    industry = st.selectbox("Industry", ["Electricity", "Cement", "Steel", "Fertilizer"])
    substance = st.selectbox("Substance", ["CO2", "CH4", "N2O"])

    dq_reliability = st.slider("DQ: Reliability Score", 0.0, 1.0, 0.6)
    dq_temporal = st.slider("DQ: Temporal Correlation", 0.0, 1.0, 0.5)
    dq_geographical = st.slider("DQ: Geographical Correlation", 0.0, 1.0, 0.5)
    dq_technological = st.slider("DQ: Technological Correlation", 0.0, 1.0, 0.5)
    dq_collection = st.slider("DQ: Data Collection Score", 0.0, 1.0, 0.5)

    margin = st.number_input("Margin of Supply Chain Emission Factors", min_value=0.0, max_value=1.0, step=0.01)

    # 🧾 Prepare input for model
    input_df = pd.DataFrame([[
        dq_reliability,
        dq_temporal,
        dq_geographical,
        dq_technological,
        dq_collection,
        margin
    ]], columns=[
        "dq_reliabilityscore_of_factors_without_margins",
        "dq_temporalcorrelation_of_factors_without_margins",
        "dq_geographicalcorrelation_of_factors_without_margins",
        "dq_technologicalcorrelation_of_factors_without_margins",
        "dq_datacollection_of_factors_without_margins",
        "margins_of_supply_chain_emission_factors"
    ])

    # 🧠 Predict
    if st.button("Predict Emissions"):
        prediction = model.predict(input_df)[0]
        st.success(f"🌍 Predicted Emission Factor (with margins): **{prediction:.4f}**")

# 🔁 Run the app
if __name__ == "__main__":
    main()


