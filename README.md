#  Customer Churn Prediction

This project predicts customer churn for a telecommunications company using a `RandomForestClassifier`. It includes **exploratory data analysis (EDA)**, **data preprocessing**, **model training**, and a **Streamlit web app** for real-time churn prediction.

---

##  Overview

- **EDA**: Analysis revealed key churn indicators like **contract type** and **monthly charges**.
- **Data Preprocessing**: Applied one-hot encoding and tenure binning.
- **Model Training**: Used `RandomForestClassifier` with handling of class imbalance.
- **Web App**: Interactive churn prediction app with modern UI.
- **Deployment**: Hosted on Streamlit Community Cloud.

---

##  Dataset

Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (7,043 records, 21 features)

### Features:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services**: `PhoneService`, `MultipleLines`, `InternetService`, etc.
- **Billing**: `Contract`, `MonthlyCharges`, `PaymentMethod`, etc.
- **Target**: `Churn` (Yes/No)

### Preprocessing:
- One-hot encoding for categorical variables.
- Binned `tenure` into six groups.
- Dropped `customerID` as non-predictive.

---

##  Project Features

| Feature                         | Description |
|----------------------------------|-------------|
|  EDA                        | Visuals and insights from customer behavior |
|  Preprocessing              | One-hot encoding, tenure binning |
|  Model Training             | RandomForestClassifier (precision ~0.50 for churn class) |
|  Streamlit App              | User-friendly UI, input validation, predictions |
|  Feature Importance         | Visualization using Random Forest importances |
|  Deployment                | Streamlit Community Cloud |

---

##  Streamlit Web App

- Input form with **15 customer features**
- Input validation with default values
- Feature importance bar chart
- Intuitive teal-based design



