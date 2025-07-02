import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("greenhouse_model.pkl")

# Streamlit Page Config
st.set_page_config(page_title="🌍 GHG Emission Predictor", layout="centered")

# Custom Styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #2E8B57;
            margin-bottom: 20px;
        }
        .box {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            color: gray;
            font-size: 13px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌍 Greenhouse Gas Emission Predictor</div>", unsafe_allow_html=True)
st.markdown("#### Powered by IBM SkillsBuild Internship | Created by Lakshmi Priya R")
st.markdown("---")

# Input Form
with st.form("prediction_form"):
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    
    industry_code = st.selectbox("🏢 Industry Code", [
        "111CA", "113FF", "211", "212", "213", "22", "23", "311FT", "313TT", "315AL", "321", "322", "323", "324", "325"
    ])
    
    industry_name = st.selectbox("🏭 Industry Name", [
        "Farms", "Forestry, fishing, and related activities", "Oil and gas extraction",
        "Mining, except oil and gas", "Support activities for mining", "Utilities", "Construction"
    ])
    
    substance = st.selectbox("💨 Gas Type", ["carbon dioxide", "methane", "nitrous oxide", "other GHGs"])
    
    submitted = st.form_submit_button("🔍 Predict Emission")

if submitted:
    input_data = pd.DataFrame({
        "Industry Code": [industry_code],
        "Industry Name": [industry_name],
        "Substance": [substance]
    })

    prediction = model.predict(input_data)[0]
    
    st.success(f"🌿 **Predicted GHG Emission Factor**: **{prediction:.3f} kg/2018 USD**")

st.markdown("<div class='footer'>© 2025 | Greenhouse GHG ML App | IBM Internship Project</div>", unsafe_allow_html=True)
