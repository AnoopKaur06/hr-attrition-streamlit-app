import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
model = pickle.load(open("attrition_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Employee Attrition Predictor")

# User input for prediction
Age = st.slider("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 50, 10)
OverTime = st.selectbox("OverTime", ["Yes", "No"])
JobSatisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)

# Convert 'OverTime' to binary (1 for Yes, 0 for No)
OverTime_bin = 1 if OverTime == "Yes" else 0

# Prepare input data for prediction
user_input = np.array([[Age, MonthlyIncome, DistanceFromHome, OverTime_bin, JobSatisfaction, YearsAtCompany]])

# Scale the input data
user_input_scaled = scaler.transform(user_input)

# Prediction button
if st.button("Predict Attrition"):
    prediction = model.predict(user_input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This employee is likely to leave.")
    else:
        st.success("✅ This employee is likely to stay.")
