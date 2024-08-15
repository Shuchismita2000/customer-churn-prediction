import streamlit as st
from home import show_home_page
from predict_page import show_predict_page
from explore_page import show_explore_page
from explore import explore_page


# Function for the About page
def show_about_page():
    st.title("📖 About")
    st.write("This project is aimed at predicting customer churn using machine learning.")
    st.write("It includes data exploration, feature engineering, model training, and prediction.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Telco Company", ["Home", "Predict", "Explore", "About"])

# Display the selected page
if page == "Home":
    show_home_page()
elif page == "Predict":
    show_predict_page()
elif page == "Explore":
    explore_page()
elif page == "About":
    show_about_page()
