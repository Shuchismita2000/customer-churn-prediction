import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
from explore import explore_page

# Load the model
#@st.cache_data(allow_output_mutation=True)


# Load the trained model encoders, scalers, and label encoders

pipeline_path = "pipeline"
scaler = {
 'Tenure Months': load(f"{pipeline_path}\Tenure Months_scaler.joblib"),
 'Monthly Charges': load(f"{pipeline_path}\Monthly Charges_scaler.joblib"),
 'CLTV': load(f"{pipeline_path}\CLTV_scaler.joblib"),

}
label_encoders = {
    'City': load(f"{pipeline_path}\City_label_encoder.joblib"),
    'Zip Code': load(f"{pipeline_path}\Zip Code_label_encoder.joblib"),
    'Internet Service': load(f"{pipeline_path}\Internet Service_label_encoder.joblib"),
    'Online Security': load(f"{pipeline_path}\Online Security_label_encoder.joblib"),
    'Tech Support': load(f"{pipeline_path}\Tech Support_label_encoder.joblib"),
    'Contract': load(f"{pipeline_path}\Contract_label_encoder.joblib"),
    'Dependents': load(f"{pipeline_path}\Dependents_label_encoder.joblib"),
}

def preprocess_data_single_input(df):
    # Select relevant features
    features = ['Zip Code', 'Dependents', 'Tenure Months', 'Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Monthly Charges', 'CLTV']
    df = df[features]

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical features using LabelEncoders
    categorical_features = ['Zip Code','Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Dependents']
    for feature in categorical_features:
        df[feature] = label_encoders[feature].transform(df[feature])
    
    # Scale numerical features
    numerical_features = ['Tenure Months', 'Monthly Charges', 'CLTV']
    for feature in numerical_features:
        df[[feature]] = scaler[feature].transform(df[[feature]])

    return df

def preprocess_data_csv(df):
    # Select relevant features
    features = ['Zip Code', 'Dependents', 'Tenure Months', 'Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Monthly Charges', 'CLTV']
    df = df[features]

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical features using LabelEncoders
    categorical_features = ['Zip Code','Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Dependents']
    for feature in categorical_features:
        try:
            df[feature] = label_encoders[feature].transform(df[feature])
        except ValueError as e:
            unseen_label = str(e).split(': ')[1]
            print(f"Warning: Unseen label '{unseen_label}' encountered. Assigning default value.")
            df[feature] = df[feature].apply(lambda x: label_encoders[feature].transform([x])[0] if x in label_encoders[feature].classes_ else np.nan)
            df[feature].fillna(label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0], inplace=True)  # Assign a default value
    
    # Scale numerical features
    numerical_features = ['Tenure Months', 'Monthly Charges', 'CLTV']
    for feature in numerical_features:
        df[[feature]] = scaler[feature].transform(df[[feature]])

    return df

def load_model():
    model = load(f"{pipeline_path}/xgb.pkl")
    return model

model = load_model()

def generate_insights(churn_status):
    if churn_status == "Churn":
        risk_level = "High"
        next_steps = "Immediate Action Required: Engage the customer with personalized offers or support."
        retention_strategy = "Consider offering discounts or loyalty programs."
        revenue_impact = "The potential loss of revenue is significant. Immediate action is required to prevent churn."
    else:
        risk_level = "Low"
        next_steps = "Monitor: Continue to monitor the customer’s activity and engagement levels."
        retention_strategy = "Consider offering a small incentive to maintain the positive relationship."
        revenue_impact = "The immediate risk to revenue is minimal, but proactive engagement can further reduce this risk."

    return risk_level, next_steps, retention_strategy, revenue_impact


def show_predict_page():
    # Title and description
    st.title('Telco Customer Churn Prediction App')
    st.markdown("""           
        **Telco Company** build this app, which uses a pre-trained model to make predictions.
              
        This app empowers you to leverage the power of machine learning to predict customer churn. By analyzing customer data, you can identify potential churners and take proactive measures to retain them.

        Key features:

        * **Data Exploration:** Gain insights into customer behavior patterns from past data.
        * **Model Prediction:** Leverage a trained model to predict churn probability.
        """)

    st.markdown("""  
                -------------------------------------------------         
        ### **Try yourself by a small example** 

        * **Single Customer:** If you have singl;e customer to predict put the values, we have settings of default values to try it out, click predict button, or you can put your values following the sample defaul values.
        * **Multiple Customer:** Leverage upload csv file option to predcit for the multiple customer, we are providing one sample csv file, you can download and upload it to check how it's working. 
        """)
    st.markdown("""  
                -------------------------------------------------         
        ### **FAQs** 
           
        f you want to know about the trends and patterns of the customers from the past data, scroll down to check the FAQs Section.
                
        """)

    st.markdown("""  
                -------------------------------------------------  """)
    
    st.write("## Prediction")
    # Option to predict for a single customer or upload a CSV file
    prediction_option = st.selectbox("Choose Prediction Type", ["Single Customer", "Upload CSV"])

    if prediction_option == "Single Customer":
    # User input for prediction
        st.write('Enter the input features:')

        # Input fields for the features based on the real dataset
        customer_id = st.text_input('CustomerID', value='7590-VHVEG')
        count = st.number_input('Count', min_value=0, value=0)
        zip_code = st.number_input('Zip Code', min_value=90001, max_value=96161, value=94109)
        city = st.text_input('City', value='San Francisco')
        gender = st.selectbox('Gender', ['Male', 'Female'], index=1)
        senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'], index=1)
        partner = st.selectbox('Partner', ['Yes', 'No'], index=0)
        dependents = st.selectbox('Dependents', ['Yes', 'No'], index=1)
        tenure_months = st.number_input('Tenure Months', min_value=0, max_value=100, value=1)
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'], index=1)
        multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'], index=2)
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], index=1)
        online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'], index=1)
        online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'], index=0)
        device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'], index=1)
        tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'], index=1)
        streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'], index=0)
        streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'], index=1)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'],index=0)
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'], index=0)
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=0)
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=29.85)
        total_charges = st.number_input('Total Charges', min_value=0.0, value=29.85)
        cltv = st.number_input('Customer Lifetime Value (CLTV)', min_value=0.0, value=1000.0)

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
            preprocessed_data = preprocess_data_single_input(input_data)
    
            # Make prediction
            prediction = model.predict(preprocessed_data)
            churn_prob = model.predict_proba(preprocessed_data)[0][1]
            # Display the results in the desired format
            churn_status = "Churn" if prediction[0] == 1 else "No Churn"
            churn_probability = f"{churn_prob:.2f}"
            st.write("### Customer Churn Prediction Report")
            st.write(f"**Customer ID:** {customer_id}") # Add the customer ID if available
            st.write(f"**Predicted Churn Status:** {churn_status}")
            st.write(f"**Churn Probability:** {churn_probability} ({churn_prob * 100:.0f}%)")

            if churn_status == "Churn":
                st.write("### **Business Insights:**")
                st.write("- **Risk Level:** **High**")
                st.write("  - The customer is currently at **high risk** of churning.")
                st.write("### **Next Steps:**")
                st.write("- **Immediate Action Required:** Engage the customer with personalized offers or support.")
                st.write("- **Retention Strategy:** Consider offering discounts or loyalty programs.")
            else:
                st.write("### **Business Insights:**")
                st.write("- **Risk Level:** **Low**")
                st.write("  - The customer is currently at **low risk** of churning.")
                st.write("### **Next Steps:**")
                st.write("- **Monitor:** Continue to monitor the customer's activity and engagement levels.")
                st.write("- **Engagement:** Consider offering a small incentive to maintain the positive relationship.")

            st.write("### **Potential Revenue Impact:**")
            if churn_status == "Churn":
                st.write("  - The potential loss of revenue is significant. Immediate action is required to prevent churn.")
            else:
                st.write("  - The immediate risk to revenue is minimal, but proactive engagement can further reduce this risk.")
    
    elif prediction_option == "Upload CSV":

        st.markdown("""
        ### 🚨 **Important Note for Users:**

        To ensure accurate predictions, please make sure that your uploaded CSV file contains the following **required columns**:

        To ensure accurate predictions, please make sure that your uploaded CSV file contains the following **required columns**:

        | **Column Name**       | **Data Type** | **Description**                                      |**Example** |
        |-----------------------|---------------|------------------------------------------------------|------------|
        |`Customer ID`         | `String`      | The unique identifier for each customer.             | '12345'     |
        | `Zip Code`            | `Integer`     | The zip code of the customer's residence.            |90210       |  
        | `Internet Service`    | `String`      | Type of internet service (e.g., DSL, Fiber optic).   |Fiber optic |
        | `Online Security`     | `String`      | Online security status (e.g., Yes, No).              |Yes     |
        | `Tech Support`        | `String`      | Tech support status (e.g., Yes, No).                 |Yes     |
        | `Contract`            | `String`      | Type of contract (e.g., Month-to-month, Two year).   |Two year     |
        | `Dependents`          | `String`      | Dependents status (e.g., Yes, No).                   |Yes     |
        | `Tenure Months`       | `Integer`     | Number of months the customer has been with the company. |24     |
        | `Monthly Charges`     | `Float`       | Amount the customer is billed each month.            |89.99     |
        | `CLTV`                | `Float`       | Customer lifetime value.                             |5000.00     
                    



        📄 **File Format**:
        - The file should be in CSV format.
        - The column names must match exactly as listed above (case-sensitive).
        - Additional columns are allowed but will be ignored in the prediction process.

    
        """)
        
        #Download the sample file 
        sample_file = "sample_customer_churn.csv"
        
        st.download_button(label="Download Sample CSV", data=open(sample_file, "rb"), file_name=sample_file, mime="csv")
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV with Customer Data", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Define the required columns
            required_columns = ['CustomerID', 'Zip Code', 'Internet Service', 'Online Security', 'Tech Support', 'Contract', 'Dependents', 'Tenure Months', 'Monthly Charges', 'CLTV']

            # Check if all required columns are present
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error("The uploaded CSV file is missing the following required columns: " + ", ".join(missing_columns))
            else:
                # Preprocess the data
                df = df[required_columns]  # Keep only the required columns for processing
                processed_data = preprocess_data_csv(df)

            # Make predictions
            predictions = model.predict(processed_data)
            churn_probs = model.predict_proba(processed_data)[:, 1]

            df['Churn Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
            df['Churn Probability'] = churn_probs

            # Generate and add business insights
            insights = df['Churn Prediction'].apply(generate_insights)
            df['Risk Level'], df['Next Steps'], df['Retention Strategy'], df['Potential Revenue Impact'] = zip(*insights)

            # Save the results to a new CSV file
            output_file = "churn_predictions.csv"
            df.to_csv(output_file, index=False)

            # Provide a download link for the output file
            st.success("Predictions made successfully! Download the results below:")
            st.download_button(label="Download CSV", data=open(output_file, "rb"), file_name=output_file, mime="text/csv")



    st.markdown("""  
                -------------------------------------------------  """)

    # Get started button
    st.write("## FAQs")
    explore_page()