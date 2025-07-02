import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load and clean dataset
df = pd.read_csv("greenhouse_gas.csv")
df = df.dropna()
df = df[df['Substance'].isin(['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])]

# Step 2: Define features and target
X = df[['Industry Name', 'Substance']]
y = df['Supply Chain Emission Factors with Margins']

# Step 3: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ['Industry Name', 'Substance'])
    ]
)

# Step 4: Full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 5: Train
pipeline.fit(X, y)

# Step 6: Save only sklearn-compatible pipeline
joblib.dump(pipeline, 'greenhouse_model.pkl')
