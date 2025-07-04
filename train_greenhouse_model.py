import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle

# Load your dataset
df = pd.read_csv("greenhouse_gas.xlsx.csv")  # or use your exact file

# Clean column names
df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

# Split input and output
X = df.drop("emission_factor", axis=1)
y = df["emission_factor"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Save as .pkl (pickle)
with open("train_greenhouse_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved successfully as train_greenhouse_model.pkl")


