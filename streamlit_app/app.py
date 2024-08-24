import streamlit as st
from predict_page import show_predict_page
from guideline import show_guideline_page
from about import about

# Function for the About page
def show_about_page():
    st.title("📖 About")
    st.write("This project is aimed at predicting customer churn using machine learning.")
    st.write("It includes data exploration, feature engineering, model training, and prediction.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Telco Company", ["Home" , "Guideline", 'About Me'])

# Display the selected page
if page == "Home":
    show_predict_page()
elif page == "Guideline":
    show_guideline_page()
elif page == "About Me":
    about()
