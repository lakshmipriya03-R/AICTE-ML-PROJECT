import os
os.system("pip install matplotlib seaborn --quiet")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Load predictions
df = pd.read_csv("predicted_emissions_output.csv")

st.title("📊 Emission Prediction Dashboard")
st.write("This dashboard shows actual vs predicted emission factors.")

# Plot
fig, ax = plt.subplots()
sns.scatterplot(x='Actual_Emission_Factor_With_Margins', y='Predicted_Emission_Factor', data=df, ax=ax)
ax.plot([df.min().min(), df.max().max()], [df.min().min(), df.max().max()], '--r')
ax.set_title("Actual vs Predicted Emission Factors")
st.pyplot(fig)

# Data preview
if st.checkbox("Show Prediction Data"):
    st.dataframe(df.head(20))
