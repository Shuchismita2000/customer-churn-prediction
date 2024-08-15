import streamlit as st

def show_home_page():
  """
  This function writes the content for the Home page.
  """

  # Title and introduction
  st.title("Customer Churn Prediction App")
  st.write("Welcome to your one-stop shop for predicting customer churn!")

  # Brief overview of the app
  st.markdown("""
  This app empowers you to leverage the power of machine learning to predict customer churn.
  By analyzing customer data, you can identify potential churners and take proactive measures to retain them.

  Key features:

  * **Data Exploration:** Gain insights into customer behavior patterns from past data.
  * **Model Prediction:** Leverage a trained model to predict churn probability.
  * **Actionable Insights:** Tailor strategies to retain valuable customers.
  """)


  

