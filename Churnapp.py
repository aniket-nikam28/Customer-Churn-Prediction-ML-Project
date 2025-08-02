import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Custom CSS for modern, compact, boxed styling with improved colors
st.markdown(
    """
    <style>
    .main {
        background-color: #f9fbfc;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 10px;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton>button {
        background-color: #26a69a;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-size: 14px;
        margin-top: 10px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00897b;
    }
    .prediction-box {
        background-color: #eceff1;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
        text-align: center;
    }
    .header {
        color: #006666;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        color: #37474f;
        font-size: 18px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    .stSidebar {
        background-color: #f9fbfc;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px;
    }
    .stSidebar .stSelectbox, .stSidebar .stNumberInput {
        margin-bottom: 5px;
    }
    .stSidebar label {
        font-size: 14px;
        margin-bottom: 2px;
        color: #37474f;
    }
    .stError {
        background-color: #ffebee;
        color: #ef5350;
        padding: 10px;
        border-radius: 5px;
    }
    .stSuccess {
        background-color: #e8f5e9;
        color: #4caf50;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
try:
    with open('model.sav', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.sav' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the expected features (exactly as in the dataset)
FEATURES = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
    'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36',
    'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
]

# Define raw categorical features and their possible values
CATEGORICAL_FEATURES = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
}

# Define numerical features and their ranges
NUMERICAL_FEATURES = {
    'SeniorCitizen': {'min': 0, 'max': 1, 'default': 0},
    'MonthlyCharges': {'min': 0.0, 'max': 200.0, 'default': 50.0},
    'TotalCharges': {'min': 0.0, 'max': 10000.0, 'default': 600.0},
    'tenure': {'min': 0, 'max': 72, 'default': 12}
}

# Function to bin tenure into tenure groups
def bin_tenure(tenure):
    if 1 <= tenure <= 12:
        return {'tenure_group_1 - 12': 1, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 0}
    elif 13 <= tenure <= 24:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 1, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 0}
    elif 25 <= tenure <= 36:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 1,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 0}
    elif 37 <= tenure <= 48:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 1, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 0}
    elif 49 <= tenure <= 60:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 1, 'tenure_group_61 - 72': 0}
    elif 61 <= tenure <= 72:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 1}
    else:
        return {'tenure_group_1 - 12': 0, 'tenure_group_13 - 24': 0, 'tenure_group_25 - 36': 0,
                'tenure_group_37 - 48': 0, 'tenure_group_49 - 60': 0, 'tenure_group_61 - 72': 0}

# Encode inputs to match model expectations
def encode_inputs(user_inputs):
    # Initialize input array with zeros
    input_data = np.zeros(len(FEATURES))
    
    # Map numerical features
    for feature in ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']:
        if feature in FEATURES:
            idx = FEATURES.index(feature)
            input_data[idx] = user_inputs[feature]
    
    # Map tenure to tenure groups
    tenure_dict = bin_tenure(user_inputs['tenure'])
    for tenure_group, value in tenure_dict.items():
        idx = FEATURES.index(tenure_group)
        input_data[idx] = value
    
    # Map categorical features to one-hot encoded columns
    for feature, value in user_inputs.items():
        if feature in CATEGORICAL_FEATURES:
            # Create the encoded feature name (e.g., 'gender_Male')
            encoded_feature = f"{feature}_{value}"
            if encoded_feature in FEATURES:
                idx = FEATURES.index(encoded_feature)
                input_data[idx] = 1
    
    return input_data.reshape(1, -1)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="subheader">Customer Details</div>', unsafe_allow_html=True)
    
    # Collect user inputs
    user_inputs = {}
    
    # Categorical inputs
    for feature, values in CATEGORICAL_FEATURES.items():
        user_inputs[feature] = st.selectbox(feature, values, key=f"select_{feature}")
    
    # Numerical inputs
    user_inputs['SeniorCitizen'] = st.selectbox('SeniorCitizen', [0, 1], key='select_SeniorCitizen')
    for feature in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        params = NUMERICAL_FEATURES[feature]
        user_inputs[feature] = st.number_input(
            feature,
            min_value=float(params['min']),
            max_value=float(params['max']),
            value=float(params['default']),
            step=0.1 if feature in ['MonthlyCharges', 'TotalCharges'] else 1.0,
            key=f"input_{feature}"
        )

# Main content
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="header">Customer Churn Prediction App</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details in the sidebar to predict churn likelihood.")
    
    # Predict button
    if st.button("Predict Churn"):
        # Get encoded input
        try:
            input_data = encode_inputs(user_inputs)
            
            # Verify input shape
            if input_data.shape[1] != len(FEATURES):
                st.error(f"Input has {input_data.shape[1]} features, but model expects {len(FEATURES)} features.")
                st.stop()
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            if prediction == 1:
                st.error(f"**Prediction: Customer is likely to CHURN** (Probability: {prediction_proba[1]:.2%})")
            else:
                st.success(f"**Prediction: Customer is NOT likely to churn** (Probability: {prediction_proba[0]:.2%})")
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Built with Streamlit | Churn Prediction Model © 2025")
    st.markdown('</div>', unsafe_allow_html=True)