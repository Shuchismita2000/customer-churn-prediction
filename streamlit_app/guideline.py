import streamlit as st
from overview import overview_page
from datacollection import data_collection_page
from datapreprocessing import data_preprocessing
from mlflow import  mlflow_page
from model import model_page

def show_guideline_page():
    st.title("Project Documentation")

    # Overview
    st.markdown("""
    Welcome to the Customer Churn Prediction project documentation. This guide will walk you through the entire process from setting up your environment to deploying the predictive model.
    """)

    # Sections for navigation
    sections = ["Overview", "1. Git & VS Code Setup", "2. Data Preprocessing", "MLflow", "3. Modeling"]
    choice = st.radio("Select a section to explore:", sections)

    if choice == "Overview":    
        overview_page()

    elif choice == "1. Git & VS Code Setup":
        data_collection_page()

    elif choice == "2. Data Preprocessing":
        data_preprocessing()

    elif choice == "MLflow":
        mlflow_page()
    
    elif choice == "3. Modeling":
        model_page()


# Separate functions for each section
def show_git_vs_code_setup():
    st.subheader("1. Git & VS Code Setup")
    st.markdown("""
    ### Step 1: Install Git
    - [Download Git](https://git-scm.com/) and follow the installation instructions.
    - Configure your Git username and email:
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
    ```

    ### Step 2: Set up VS Code
    - [Download VS Code](https://code.visualstudio.com/).
    - Install necessary extensions like Python, GitLens, etc.
    """)

def show_data_preprocessing():
    st.subheader("2. Data Preprocessing")
    st.markdown("""
    ### Feature Engineering
    - Understand the data and create new features if necessary.
    - Normalize or scale features as required.

    ### Feature Selection
    - Select the most relevant features based on importance or correlation with the target variable.
    - Use techniques like recursive feature elimination or Lasso for feature selection.
    """)

def show_modeling():
    st.subheader("3. Modeling (Including MLflow)")
    st.markdown("""
    ### Model Building
    - Train your model using algorithms like Random Forest, XGBoost, etc.
    - Evaluate model performance using metrics like accuracy, precision, recall, etc.

    ### MLflow Integration
    - Track your experiments and model parameters using MLflow.
    - Save and deploy models with MLflow's model registry.
    """)

# Call the guideline page in your main app
if __name__ == "__main__":
    show_guideline_page()
