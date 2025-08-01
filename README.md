# Customer Churn Prediction

Overview
This project predicts customer churn for a telecommunications company using a RandomForestClassifier. It encompasses exploratory data analysis (EDA), model training, and an interactive Streamlit web app for real-time churn predictions. The app features input validation, a modern UI with vibrant colors, and a visualization of key features influencing churn.
The dataset used is the Telco Customer Churn dataset, which includes customer demographics, services, and billing information. The project demonstrates skills in data preprocessing, machine learning, and web app development.
Features

Exploratory Data Analysis (EDA): Analyzed the Telco Customer Churn dataset to identify patterns (e.g., higher churn with month-to-month contracts) using Pandas, Seaborn, and Matplotlib. See notebooks/Churn_EDA_Analysis.ipynb.
Data Preprocessing: Converted categorical variables to one-hot encoded features and binned tenure into groups (e.g., 1-12, 13-24 months), resulting in 50 features. Preprocessed data saved as data/tel_churn.csv.
Model Training: Trained a RandomForestClassifier to predict churn, handling imbalanced data and achieving reasonable performance (precision ~0.50 for churn class).
Web App: Built a Streamlit app (app/churn_prediction_gui.py) for interactive predictions, featuring:
Input forms for 15 raw features (e.g., tenure, Contract, MonthlyCharges).
Input validation to ensure values are within acceptable ranges.
Feature importance visualization to highlight key predictors (e.g., Contract, tenure).
Modern UI with vibrant teal buttons and soft off-white backgrounds.


Deployment: Deployed on Streamlit Community Cloud for public access.

Dataset
The dataset (data/WA_Fn-UseC_-Telco-Customer-Churn.csv) contains 7,043 records with 21 features, including:

Demographics: gender, SeniorCitizen, Partner, Dependents
Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Billing: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
Target: Churn (Yes/No)

Key preprocessing steps:

One-hot encoded categorical variables (e.g., Contract_Month-to-month, Contract_One year).
Binned tenure into six groups (1-12, 13-24, 25-36, 37-48, 49-60, 61-72 months).
Dropped customerID as it’s non-predictive.

Installation

Clone the Repository:
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction


Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
streamlit==1.29.0
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2


Ensure Model File:

Place model.sav (trained RandomForestClassifier) in the model/ directory.


Download Dataset (optional):

The original dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) and preprocessed data (tel_churn.csv) are in the data/ directory. If missing, download from Kaggle.



Usage

Run the Streamlit App:
streamlit run app/churn_prediction_gui.py


Access the App:

Open your browser at http://localhost:8501.
Enter customer details in the sidebar (e.g., tenure, Contract, MonthlyCharges).
Click "Predict Churn" to view the prediction and feature importance plot.


Explore EDA:

Open notebooks/Churn_EDA_Analysis.ipynb in Jupyter Notebook to review data analysis and visualization








                   

Deployment
The app is deployed on Streamlit Community Cloud





