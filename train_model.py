import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load and clean data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df = df[["Age", "MonthlyIncome", "DistanceFromHome", "OverTime", "JobSatisfaction", "YearsAtCompany", "Attrition"]]
df = df.dropna()

# Encode OverTime and Attrition
df["OverTime"] = df["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
df["Attrition"] = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

X = df[["Age", "MonthlyIncome", "DistanceFromHome", "OverTime", "JobSatisfaction", "YearsAtCompany"]]
y = df["Attrition"]

# Apply SMOTE to fix imbalance
sm = SMOTE(sampling_strategy=0.6, random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y_resampled)

# Save model and scaler
with open("attrition_model1.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
