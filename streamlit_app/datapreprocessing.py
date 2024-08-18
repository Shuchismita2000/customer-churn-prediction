import streamlit as st

def data_preprocessing():
    st.title("Data Preprocessing")
    
    st.write("""
    Feature engineering is a critical step in preparing the data for modeling. It involves transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model performance. In this section, we’ll create new features and transform existing ones based on the insights we gathered during EDA.
    """)

    st.header("Step 1: Load the Data")
    st.write("""
    Let's start by loading the data and getting a feel for what we're working with.
    """)

    st.subheader("Load the Dataset")
    st.write("""
    First, let's load the Telco Customer Churn dataset into a Pandas DataFrame. Create a new Jupyter Notebook to perform this task.
    ```python
    # Example code to load the dataset
    import pandas as pd
    df = pd.read_csv('Telco-Customer-Churn.csv')
    ```
    """)
    
    st.subheader("Basic Information")
    st.write("""
    Check the basic information of the dataset to understand its structure.
    ```python
    df.info()
    df.describe()
    ```
    """)

    st.header("Step 2: Handle Missing Values")
    st.write("""
    Data often comes with some missing values. Let's identify and handle them to ensure our analysis is accurate.
    """)

    st.subheader("Identify Missing Values")
    st.write("""
    Identify columns with missing values.
    ```python
    missing_values = df.isnull().sum()
    print(missing_values)
    ```
    """)

    st.subheader("Handle Missing Values")
    st.write("""
    For simplicity, we'll drop rows with missing values.
    ```python
    df = df.dropna()
    ```
    """)

    st.header("Step 3: Data Cleaning")
    st.write("""
    Clean data is the foundation of good analysis. Let's clean and transform our data.
    """)

    st.subheader("Dropping Unnecessary Columns")
    st.write("""
    Drop columns that are not needed for the analysis.
    ```python
    df = df.drop(['customerID'], axis=1)
    ```
    """)

    st.subheader("Converting Columns")
    st.write("""
    Convert data types of certain columns if necessary.
    ```python
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    ```
    """)

    st.subheader("Check for Duplicates")
    st.write("""
    Remove any duplicate rows if present.
    ```python
    df = df.drop_duplicates()
    ```
    """)
#---------------------------------------------------------------------------------------
    st.header("Step 1: Encode Categorical Variables")
    st.write("""
    Let's encode categorical variables by applying LabelEncoder. This will allow our models to interpret these variables properly.
    
    The next step in our data preprocessing pipeline is to encode the categorical variables. This step is crucial because machine learning models typically require numerical input, and categorical data must be converted into a format that the models can interpret. Common techniques include one-hot encoding and label encoding, which help transform categorical variables into a numerical form, ensuring that the model correctly understands and utilizes these features during training.         
     """)
   
    st.write("""
    ```python
    # List of categorical columns in the dataset
categorical_columns = [
    'City', 'Zip Code', 'Gender', 'Senior Citizen', 'Partner', 'Dependents', 
    'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 
    'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 
    'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method', 
    'Churn Reason', 'Churn Label'
]

# Import the LabelEncoder from sklearn for encoding categorical variables
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# Dictionary to hold LabelEncoders for each categorical column
label_encoders = {}

# Loop through each categorical column and apply LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()  # Initialize the LabelEncoder for the column
    df[col] = le.fit_transform(df[col])  # Encode the column and replace it in the dataframe
    label_encoders[col] = le  # Store the fitted LabelEncoder for future use

# Save each LabelEncoder to disk for use during inference
for column, le in label_encoders.items():  
    # Save the LabelEncoder as a .joblib file in the specified directory
    dump(le, rf'..\customer-churn-prediction\pipeline\{column}_label_encoder.joblib')

    """)




    st.header("Step 3: Scale Numerical Features")
    st.write("""
    After encoding the categorical variables, the next step is to scale the numerical features. Scaling is essential to ensure that all numerical features are on a similar scale, which can significantly improve model performance, especially for algorithms that rely on distance metrics.

    **Standardize Numerical Features:** In this step, we'll use StandardScaler to standardize the numerical features. Standardization transforms the data to have a mean of 0 and a standard deviation of 1, making the features comparable and stabilizing the learning process for many machine learning models.
    """)
    
    st.write("""
     ```python
    # Import the StandardScaler from sklearn for scaling numerical features
from sklearn.preprocessing import StandardScaler
from joblib import dump

# List of numerical features that need to be standardized
numerical_features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']

# Loop through each numerical feature to apply StandardScaler
for feature in numerical_features:
    scaler = StandardScaler()  # Initialize the StandardScaler for the feature
    
    # Fit the scaler to the feature data and transform it
    df[[feature]] = scaler.fit_transform(df[[feature]])
    
    # Save the fitted scaler as a .joblib file for later use
    dump(scaler, rf'..\customer-churn-prediction\pipeline\{feature}_scaler.joblib')

# If needed, you can scale all numerical features at once using a single scaler
# This line demonstrates how to do that for all numerical features together
df[numerical_features] = scaler.fit_transform(df[numerical_features])

    """)
    
    
    
    st.header("Step 5: Feature Selection")
    st.write("""
    The next step is to focus on selecting the most relevant features. This process not only simplifies the model but can also enhance its performance by removing irrelevant or redundant data.

    **Feature Selection:** Feature selection techniques are applied to identify the most significant features that contribute to the model's predictive power. By filtering out less important features, we can reduce overfitting, speed up training time, and improve the model's generalization to new data.
     """)
    
    st.write("""
    ```python
             # Drop unnecessary columns that won't be used in the model
df.drop(['Unnamed: 0', 'CustomerID', 'Count', 'Churn Reason', 'Churn Value', 'Churn Score'], axis=1, inplace=True)

# Separate the features (X) from the target variable (y)
X = df.drop('Churn Label', axis=1)  # Features
y = df['Churn Label']  # Target variable
             
    """)

    st.subheader("1. Correlation Analysis")
    st.write("""             
    The first step in feature selection is to perform a correlation analysis. This helps in identifying and removing highly correlated features, which can lead to multicollinearity in your model. Multicollinearity occurs when two or more features are highly correlated, potentially skewing the model's predictions and making it difficult to determine the individual effect of each feature. By removing one of the correlated features, we can simplify the model and improve its performance.
     """)
    st.write("""
    ```python
        # Calculate the correlation matrix to identify relationships between features
corr_matrix = X.corr().abs()  # Use absolute values to focus on the strength of the correlations
corr_matrix

import numpy as np

# Select the upper triangle of the correlation matrix
# This avoids redundant comparisons by focusing on pairs of features
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Identify features with a correlation greater than 0.8
# These are likely to be highly correlated and may introduce multicollinearity
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop the highly correlated features from the dataset
X = X.drop(columns=to_drop)  
                      
    """)
    
    st.subheader("2. Recursive Feature Elimination (RFE)")
    st.write("""
    Recursive Feature Elimination (RFE) is a powerful technique used to select the most important features for your model. By recursively removing the least important features, RFE helps in narrowing down the feature set to those that contribute the most to the model's predictive power. This process is done in conjunction with a chosen model, where the model's importance scores guide the elimination process. RFE can significantly enhance model performance by focusing on the most relevant features while reducing noise and complexity.
     """)
    
    st.write("""
    ```python 
             # Import RFE (Recursive Feature Elimination) and RandomForestClassifier for feature selection
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Instantiate a RandomForestClassifier as the model for RFE
model = RandomForestClassifier()

# Instantiate RFE with the chosen model and the desired number of features to select
rfe = RFE(estimator=model, n_features_to_select=10)  # Select the top 10 features

# Fit RFE to the data (X and y) to perform feature selection
rfe.fit(X, y)

# Select the features that were identified as important by RFE
X_rfe = X.loc[:, rfe.support_]  # Keep only the features selected by RFE
             
             
    """)
    

    st.header("Step 4: Handle Class Imbalance")
    st.write("""
    In cases where the target variable is skewed, addressing class imbalance is crucial to improving model performance. Imbalanced classes can lead to biased models that perform well on the majority class but poorly on the minority class.

    **Balance Classes Using SMOTE:** One effective method to handle class imbalance is the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic samples for the minority class by interpolating between existing samples, helping to create a more balanced dataset. By doing so, it allows the model to learn from both classes more effectively, leading to better generalization and improved predictive accuracy, particularly for the minority class.
    """)
    
    st.write("""
    
    ```python
    # Check the distribution of the target variable to understand the class imbalance
    df['Churn Label'].value_counts() #0:5174  1:1869

    # Import SMOTE (Synthetic Minority Over-sampling Technique) from imbalanced-learn
    from imblearn.over_sampling import SMOTE

    # Instantiate SMOTE to handle class imbalance
    smote = SMOTE()

    # Apply SMOTE to the data
    # This generates synthetic samples for the minority class to balance the dataset
    X_resampled, y_resampled = smote.fit_resample(X_rfe, y)

    # Check the shape of the resampled data to confirm the class balancing
    X_resampled.shape, y_resampled.shape   #((10348, 10), (10348,))

    """)


    st.header("Summary of Feature Engineering")
    st.write("""
    We've crafted and transformed features that will feed into our model. Here's a summary of our feature engineering process:
    
    - **Interaction Features:** Created interaction features between key variables.
    - **Categorical Encoding:** Applied one-hot encoding to categorical variables.
    - **Scaling:** Standardized numerical features to ensure they are on a similar scale.
    - **Class Imbalance Handling:** Used SMOTE to balance the classes in the target variable.
    - **Feature Selection:** Selected the top features based on their importance.
    """)


