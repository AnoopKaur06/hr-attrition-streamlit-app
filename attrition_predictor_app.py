import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model (you'll save this from your previous notebook)
model = pickle.load(open("attrition_model.pkl", "rb"))

st.title("Employee Attrition Prediction")

# Collect user input
Age = st.slider("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
DistanceFromHome = st.slider("Distance From Home (km)", 1, 50, 10)
OverTime = st.selectbox("Works Overtime?", ["Yes", "No"])
JobSatisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)

# Process input into model format
OverTime_bin = 1 if OverTime == "Yes" else 0

# Create input for model
input_data = np.array([[Age, MonthlyIncome, DistanceFromHome, OverTime_bin, JobSatisfaction, YearsAtCompany]])

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("This employee is likely to leave.")
    else:
        st.success("This employee is likely to stay.")
