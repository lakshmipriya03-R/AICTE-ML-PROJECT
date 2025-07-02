import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("greenhouse_gas.csv")

# Basic cleanup
df = df.dropna()
df = df[df['Substance'].isin(['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])]

# Features and target
X = df[["Industry Name", "Substance"]]
y = df["Supply Chain Emission Factors with Margins"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Industry Name", "Substance"])
    ]
)

# Final pipeline
model_pipeline = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X, y)

# Save model — NO custom class used!
joblib.dump(model_pipeline, "greenhouse_model.pkl")
