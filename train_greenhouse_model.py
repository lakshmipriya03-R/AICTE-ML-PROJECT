# train_greenhouse_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load and clean data
df = pd.read_csv("greenhouse_gas.csv")
df = df.dropna()
X = df[["Industry Code", "Industry Name", "Substance"]]
y = df["Supply Chain Emission Factors with Margins"]

# Preprocessing
categorical_cols = ["Industry Code", "Industry Name", "Substance"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "greenhouse_model.pkl")

# Optional: Save feature names if needed for UI
feature_names = preprocessor.fit(X).get_feature_names_out()
joblib.dump(feature_names, "model_features.pkl")

print("✅ Model trained and saved successfully.")
