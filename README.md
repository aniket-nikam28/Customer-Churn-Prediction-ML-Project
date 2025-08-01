# Customer Churn Prediction

## Overview
This project predicts customer churn for a telecommunications company using a RandomForestClassifier. It includes exploratory data analysis (EDA), model training, and an interactive Streamlit web app for real-time churn predictions. The app features input validation, a modern UI with vibrant styling, and a visualization of key features influencing churn predictions.
The project leverages the Telco Customer Churn dataset to analyze customer behavior and predict churn, demonstrating skills in data preprocessing, machine learning, and web development.
## Features

Exploratory Data Analysis (EDA): Conducted in notebooks/Churn_EDA_Analysis.ipynb using Pandas, Seaborn, and Matplotlib to uncover patterns (e.g., higher churn rates for month-to-month contracts).
Data Preprocessing: Transformed categorical variables into one-hot encoded features and binned tenure into six groups (1-12, 13-24, 25-36, 37-48, 49-60, 61-72 months), resulting in 50 features. Preprocessed data saved as data/tel_churn.csv.
Model Training: Trained a RandomForestClassifier (model.sav) to predict churn, handling imbalanced data with moderate precision (~0.50 for churn class).
Streamlit Web App: Built in app/churn_prediction_gui.py, featuring:
User-friendly input forms for 15 raw features (e.g., tenure, Contract, MonthlyCharges).
Input validation to ensure valid ranges.
Feature importance plot highlighting key predictors (e.g., Contract, tenure).
Modern UI with teal buttons, off-white backgrounds, and subtle animations.


Deployment: Hosted on Streamlit Community Cloud for public access.

##Dataset
The Telco Customer Churn dataset (data/WA_Fn-UseC_-Telco-Customer-Churn.csv) contains 7,043 records and 21 features:

Demographics: gender, SeniorCitizen, Partner, Dependents
Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Billing: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Target: Churn (Yes/No)

##Preprocessing:

One-hot encoded categorical variables (e.g., Contract_Month-to-month, InternetService_Fiber optic).
Binned tenure into groups for model training.
Dropped non-predictive customerID.




##Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
streamlit==1.29.0
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2


##Ensure Model File:

Place the trained model (model.sav) in the model/ directory.


##Download Dataset (optional):

The dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) and preprocessed data (tel_churn.csv) are in data/. If missing, download from Kaggle.



