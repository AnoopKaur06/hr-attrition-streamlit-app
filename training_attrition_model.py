import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Select the relevant features for training
X = data[["Age", "MonthlyIncome", "DistanceFromHome", "OverTime", "JobSatisfaction", "YearsAtCompany"]].copy()

# Convert categorical 'OverTime' to numeric (1 for Yes, 0 for No)
X["OverTime"] = X["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)

# Target variable (Attrition)
y = data["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.6, random_state=42)  # Resampling ratio 0.6
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Save the trained model and scaler to pickle files
with open("attrition_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
