import streamlit as st
import pickle
import numpy as np

# ✅ Load model
with open("greenhouse_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Title
st.title("🔮 Predict Emission Factor With Margins")
st.markdown("Give the input values below to predict emission factor 🔥")

# ✅ Inputs from user
industry_code = st.selectbox("Industry Code", [101, 102, 103, 104])
substance = st.selectbox("Substance", [1, 2, 3])
dq_reliability = st.slider("DQ Reliability", 0.0, 1.0, 0.9)
dq_temporal = st.slider("DQ Temporal Correlation", 0.0, 1.0, 0.7)
dq_geographical = st.slider("DQ Geographical Correlation", 0.0, 1.0, 0.8)
dq_technological = st.slider("DQ Technological Correlation", 0.0, 1.0, 0.85)
dq_datacollection = st.slider("DQ Data Collection", 0.0, 1.0, 0.95)
margin = st.number_input("Supply Chain Margin", value=0.01)

# ✅ Predict Button
if st.button("Predict Emission Factor"):
    input_data = np.array([[industry_code, substance,
                            dq_reliability, dq_temporal, dq_geographical,
                            dq_technological, dq_datacollection, margin]])
    prediction = model.predict(input_data)[0]
    st.success(f"📈 Predicted Emission Factor with Margin: **{prediction:.4f}**")

