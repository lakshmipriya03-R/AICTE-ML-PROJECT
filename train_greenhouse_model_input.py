import pandas as pd
import streamlit as st
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ✅ Step 1: Define your dataset directly in the code
data = {
    "industry": [1, 2, 3, 4, 5, 1, 2, 3],
    "substance": [1, 2, 3, 4, 5, 3, 2, 1],
    "dq_marginal": [0.8, 0.6, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9],
    "dq_temporal": [0.9, 0.7, 0.6, 0.8, 0.7, 0.9, 0.7, 0.6],
    "dq_geographical": [0.85, 0.65, 0.75, 0.55, 0.9, 0.85, 0.65, 0.75],
    "emission_factor": [100, 200, 150, 250, 300, 120, 210, 130]
}

df = pd.DataFrame(data)

# ✅ Step 2: Train model
X = df.drop("emission_factor", axis=1)
y = df["emission_factor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

# ✅ Step 3: Streamlit UI
st.set_page_config(page_title="GHG Predictor")
st.title("🌍 Greenhouse Gas Emission Predictor")

st.sidebar.header("Enter Input Values")
industry = st.sidebar.slider("Industry", 1, 5, 3)
substance = st.sidebar.slider("Substance", 1, 5, 3)
dq_marginal = st.sidebar.slider("DQ Marginal", 0.0, 1.0, 0.8)
dq_temporal = st.sidebar.slider("DQ Temporal", 0.0, 1.0, 0.7)
dq_geographical = st.sidebar.slider("DQ Geographical", 0.0, 1.0, 0.75)

input_df = pd.DataFrame({
    "industry": [industry],
    "substance": [substance],
    "dq_marginal": [dq_marginal],
    "dq_temporal": [dq_temporal],
    "dq_geographical": [dq_geographical]
})

prediction = model.predict(input_df)[0]
st.subheader("📊 Predicted Emission Factor")
st.success(f"{prediction:.2f}")


