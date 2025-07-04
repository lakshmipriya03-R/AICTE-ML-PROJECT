{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "15a57d15-0aba-4ab2-bc89-f919f264725a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model saved successfully as train_greenhouse_model.json\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from xgboost import XGBRegressor\n",
    "\n",
    "# Define a small sample dataset\n",
    "data = pd.DataFrame({\n",
    "    'industry': ['Electricity', 'Cement', 'Transport', 'Fertilizer', 'Electricity'],\n",
    "    'ghg': ['CO2', 'CH4', 'N2O', 'CO2', 'CH4'],\n",
    "    'margin': [1.2, 2.3, 1.8, 3.1, 2.0],\n",
    "    'dq_assessment': [0.9, 0.7, 0.8, 0.6, 0.7],\n",
    "    'dq_coverage': [0.85, 0.75, 0.8, 0.65, 0.7],\n",
    "    'dq_uncertainty': [0.8, 0.6, 0.7, 0.6, 0.65],\n",
    "    'dq_verification': [0.95, 0.85, 0.9, 0.8, 0.85],\n",
    "    'emission_factor': [10.5, 12.3, 9.8, 13.5, 11.0]\n",
    "})\n",
    "\n",
    "# Preprocess\n",
    "X = data.drop(\"emission_factor\", axis=1)\n",
    "y = data[\"emission_factor\"]\n",
    "X = pd.get_dummies(X)\n",
    "\n",
    "# Train model\n",
    "model = XGBRegressor()\n",
    "model.fit(X, y)\n",
    "\n",
    "# Save model as JSON\n",
    "model.save_model(\"train_greenhouse_model.json\")\n",
    "\n",
    "print(\"✅ Model saved successfully as train_greenhouse_model.json\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "357f1ea6-c4be-492c-bd33-04407a61950d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
