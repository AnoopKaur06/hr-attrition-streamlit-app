﻿# hr-attrition-streamlit-app
This project is a machine learning-based web app that predicts whether an employee is likely to leave the organization based on six key HR features. It uses a Random Forest classifier trained on the IBM HR Analytics dataset and deployed using Streamlit.

## 🔍 Features
- Built with Streamlit for interactive UI
- Uses only 6 input features: `Age`, `MonthlyIncome`, `DistanceFromHome`, `OverTime`, `JobSatisfaction`, `YearsAtCompany`
- Trained using Random Forest and SMOTE for data balancing
- Includes feature scaling for better model performance

## 📁 Project Structure
├── attrition_model.pkl ├── scaler.pkl ├── app.py ├── README.md ├── requirements.txt └── WA_Fn-UseC_-HR-Employee-Attrition.csv

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
