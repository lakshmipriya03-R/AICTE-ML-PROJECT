import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page config
st.set_page_config(page_title="🌱 GHG Emission Predictor", layout="centered")

st.markdown("<h1 style='text-align:center;color:#007200;'>🌍 Greenhouse Gas Emission Predictor</h1>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("greenhouse_gas.csv")
    df = df.dropna()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df = load_data()

# Select columns
feature_cols = ['Industry Name', 'Substance', 'Supply Chain Emission Factors without Margins']
target_col = 'Supply Chain Emission Factors with Margins'

# Sidebar Info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=180)
    st.markdown("### 👩‍💻 Lakshmi Priya R")
    st.markdown("🧠 AI & Data Science Intern at IBM SkillsBuild")
    st.markdown("---")
    st.info("Fill the details to predict greenhouse gas emissions")

# User Inputs
with st.form("ghg_form"):
    industry = st.selectbox("🏭 Industry", sorted(df['Industry Name'].unique()))
    substance = st.selectbox("🧪 Gas Type", sorted(df['Substance'].unique()))
    base_emission = st.number_input("🔢 Base Emission (kg per USD)", min_value=0.0, max_value=5.0, value=0.5)

    submit = st.form_submit_button("🚀 Predict")

if submit:
    # Prepare training data
    X = df[feature_cols]
    y = df[target_col]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    categorical_features = ['Industry Name', 'Substance']
    numeric_features = ['Supply Chain Emission Factors without Margins']

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder='passthrough')

    # Pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    user_df = pd.DataFrame({
        'Industry Name': [industry],
        'Substance': [substance],
        'Supply Chain Emission Factors without Margins': [base_emission]
    })

    prediction = model.predict(user_df)[0]

    # Show result
    st.success(f"🌡️ Predicted GHG Emission with Margin: **{prediction:.3f} kg CO₂e/USD**")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center>🌱 IBM SkillsBuild Internship • Greenhouse Gas Emission Project</center>", unsafe_allow_html=True)

