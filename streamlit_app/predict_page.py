import streamlit as st
from joblib import load
import numpy as np
import pandas as pd


# Load the model
#@st.cache_data(allow_output_mutation=True)


# Load the trained model encoders, scalers, and label encoders
scaler = load('pipeline/scaler.joblib')
label_encoders = {
    'Zip Code': load('pipeline/Zip Code_label_encoder.joblib'),
    'Internet Service': load('pipeline/Internet Service_label_encoder.joblib'),
    'Online Security': load('pipeline/Online Security_label_encoder.joblib'),
    'Tech Support': load('pipeline/Tech Support_label_encoder.joblib'),
    'Contract': load('pipeline/Contract_label_encoder.joblib'),
    'Dependents': load('pipeline/Dependents_label_encoder.joblib'),
}

def preprocess_data(df):
    # Select relevant features
    features = ['Zip Code', 'Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Dependents', 'Tenure Months', 'Monthly Charges', 'CLTV']
    df = df[features]

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical features using LabelEncoders
    categorical_features = ['Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Dependents', 'Zip Code']
    for feature in categorical_features:
        df[feature] = label_encoders[feature].transform(df[feature])
    
    # Scale numerical features
    numerical_features = ['Tenure Months', 'Monthly Charges', 'CLTV']
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

def load_model():
    model = load('pipeline/xgb.pkl')
    return model

model = load_model()

def show_predict_page():
    # Title and description
    st.title('Customer Churn Prediction App')
    st.write('This app uses a pre-trained model to make predictions.')

    # User input for prediction
    st.write('Enter the input features:')

    # Input fields for the features based on the real dataset
    customer_id = st.text_input('CustomerID')
    count = st.number_input('Count', min_value=0, value=0)
    zip_code = st.number_input('Zip Code', min_value=90001, max_value=96161, value=90001)
    city = st.text_input('City')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure_months = st.number_input('Tenure Months', min_value=0, max_value=100, value=0)
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, value=0.0)
    cltv = st.number_input('Customer Lifetime Value (CLTV)', min_value=0.0, value=0.0)

    # Create a dataframe from user inputs
    input_data = pd.DataFrame({
    'CustomerID': [customer_id],
    'Count': [count],
    'Zip Code': [zip_code],
    'City': [city],
    'Gender': [gender],
    'Senior Citizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'Tenure Months': [tenure_months],
    'Phone Service': [phone_service],
    'Multiple Lines': [multiple_lines],
    'Internet Service': [internet_service],
    'Online Security': [online_security],
    'Online Backup': [online_backup],
    'Device Protection': [device_protection],
    'Tech Support': [tech_support],
    'Streaming TV': [streaming_tv],
    'Streaming Movies': [streaming_movies],
    'Contract': [contract],
    'Paperless Billing': [paperless_billing],
    'Payment Method': [payment_method],
    'Monthly Charges': [monthly_charges],
    'Total Charges': [total_charges],
    'CLTV': [cltv]
    })

    # Button for prediction
    if st.button('Predict'):
    # Preprocess the input data
        preprocessed_data = preprocess_data(input_data)
    
        # Make prediction
        prediction = model.predict(preprocessed_data)
        churn_prob = model.predict_proba(preprocessed_data)[0][1]
        st.write(f'The predicted churn status is: {"Churn" if prediction[0] == 1 else "No Churn"}')
        st.write(f'The probability of churn is: {churn_prob:.2f}')