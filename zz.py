import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from prophet import Prophet
from xgboost import XGBRegressor, plot_importance
import folium
from streamlit_folium import st_folium

# Load Data
df = pd.read_excel("greenhouse_gas.xlsx")

# Title
st.title("🌍 Advanced Greenhouse Gas Emissions Analyzer")

# Show raw data
if st.checkbox("📂 Show Raw Data"):
    st.dataframe(df)

# -----------------------------------------------
# ML MODEL & FEATURE IMPORTANCE
# -----------------------------------------------
target = 'supply_chain_emission_factors_with_margins'
features = df.select_dtypes(include=[np.number]).drop(columns=[target]).columns

model = XGBRegressor()
model.fit(df[features], df[target])
preds = model.predict(df[features])

# Feature Importance
st.subheader("🎯 Feature Importance")
fig, ax = plt.subplots(figsize=(10, 5))
plot_importance(model, importance_type='gain', max_num_features=10, ax=ax)
st.pyplot(fig)

# Prediction vs Actual
st.subheader("📈 Actual vs Predicted")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df[target], y=preds, ax=ax2)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
st.pyplot(fig2)

# -----------------------------------------------
# SHAP EXPLAINABILITY
# -----------------------------------------------
st.subheader("🧠 SHAP Explanation")
explainer = shap.Explainer(model, df[features])
shap_values = explainer(df[features])

fig3, ax3 = plt.subplots()
shap.summary_plot(shap_values, df[features], plot_type="bar", show=False)
st.pyplot(fig3)

# -----------------------------------------------
# FORECASTING WITH PROPHET
# -----------------------------------------------
st.subheader("🔮 Emission Forecasting")

# Create synthetic timeline if 'year' not available
date_range = pd.date_range(start='2000', periods=len(df), freq='Y')
forecast_df = pd.DataFrame({'ds': date_range, 'y': df[target]})

m = Prophet()
m.fit(forecast_df)
future = m.make_future_dataframe(periods=5, freq='Y')
forecast = m.predict(future)

fig4 = m.plot(forecast)
st.pyplot(fig4)

# -----------------------------------------------
# GEO-VISUALIZATION
# -----------------------------------------------
st.subheader("🗺️ Industry-wise Emission Map")

# Fake coordinates for demonstration
industry_coords = {
    "Electricity": [28.6, 77.2],    # Delhi
    "Cement": [23.0, 72.6],         # Ahmedabad
    "Steel": [22.6, 88.4],          # Kolkata
    "Fertilizer": [26.8, 80.9],     # Lucknow
    "Petroleum": [19.0, 72.8],      # Mumbai
}

map_ = folium.Map(location=[23.5, 80.0], zoom_start=5)
for industry, coords in industry_coords.items():
    avg = df[df['industry_name'].str.contains(industry, case=False, na=False)][target].mean()
    folium.CircleMarker(
        location=coords,
        radius=7,
        popup=f"{industry}: {avg:.2f} kg CO₂ eq",
        color='crimson',
        fill=True,
        fill_color='crimson'
    ).add_to(map_)

st_folium(map_, width=700)

# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.markdown("💡 *Made with love to stand out among 15,000+ peers. Fully explainable, forecastable, and visualized.*")

