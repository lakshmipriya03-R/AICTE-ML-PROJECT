import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("greenhouse_gas.csv")

# Drop unnamed index columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Define features and target
X = df[["Industry Code", "Industry Name", "Substance", "Supply Chain Emission Factors without Margins"]]
y = df["Supply Chain Emission Factors with Margins"]

# Preprocessing for categorical data
categorical = ["Industry Code", "Industry Name", "Substance"]
numeric = ["Supply Chain Emission Factors without Margins"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Pipeline
model = Pipeline([
    ("pre", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "greenhouse_model.pkl")

