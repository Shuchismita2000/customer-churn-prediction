# Customer Churn Prediction

This project aims to predict customer churn using machine learning models. The web application is built using Streamlit and is deployed on Render.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
Customer churn is a critical issue for businesses as retaining existing customers is often more cost-effective than acquiring new ones. This project provides a streamlined solution for predicting whether a customer will churn based on various features. The app allows users to explore the data, visualize trends, and generate predictions.

## Features
- **Interactive Data Visualization**: Explore customer data with interactive charts and graphs.
- **Prediction Model**: Predict customer churn based on key features.
- **Scalability**: Deployed on Render, ensuring that the app can handle multiple requests concurrently.

## Demo
You can view the deployed application here: [Customer Churn Prediction App](https://customer-churn-prediction-427t.onrender.com/)

## Technologies Used
- **Python**: The core programming language used for building the machine learning model.
- **Streamlit**: Framework used to create the web interface.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Joblib**: To save and load the machine learning models and preprocessors.
- **Render**: For deployment.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shuchismita2000/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install the required packages:**

 ```bash
pip install -r requirements.txt
```
3. **Run the application:**

```bash
streamlit run streamlit_app/app.py
```
4. File Structure
5. 
customer-churn-prediction/
│
├── streamlit_app/
│   ├── app.py                # Main app script
│   ├── predict_page.py        # Prediction page logic
│   ├── explore.py             # Data exploration page
│   ├── guideline.py           # Guidelines and instructions
│   └── overview.py      # Overview page logic
│   └── datacollection.py      # Data collection page logic
│   └── datapreprocessing.py      # Data preprocessing page logic
│   └── mlflow.py      # MLflow page logic
│   └── model.py      # Model page logic
│
├── pipeline/
│   ├── Tenure Months_scaler.joblib  # Scalers 
│   └── model.joblib                # Trained machine learning model
│
├── Telco_customer_churn.xlsx    # Dataset
├── Sample_customer_churn.xlsx    # Dataset
│
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
Contributing
Contributions are welcome! Please create an issue or a pull request for any improvements or suggestions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

csharp
Copy code

You can copy and paste this into your `README.md` file in your GitHub repository.





