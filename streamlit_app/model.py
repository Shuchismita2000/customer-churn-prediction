import streamlit as st

def model_page():

    st.title("Modeling (Integrating MLflow)")


    st.write("""
    Now that we've set up MLflow, it's time to integrate it into our existing modeling code. By doing so, we can effectively track our experiments, log important metrics, and visualize the results using the MLflow UI.

    In this section of the blog, we'll guide you through the process of embedding MLflow into your machine learning workflow. This integration will enable you to monitor the performance of different models, compare various runs, and easily manage the lifecycle of your models—all from a single platform. By the end, you'll have a clear understanding of how MLflow enhances your ability to track, reproduce, and optimize your machine learning experiments.
    """)


    st.header("Split the Data")
    st.write("""
Before we start modeling, it's essential to split the data into training and testing sets.

**Train-Test Split:** Dividing the dataset into training and testing sets ensures that we can evaluate the performance of our model on unseen data. The training set is used to build and train the model, while the testing set is reserved for assessing how well the model generalizes to new, unseen data. This split helps in preventing overfitting and provides a more accurate measure of the model's performance.""")
    
    st.write("""
    ```python 
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    """)

    st.header("Setting the import and mlflow experiment")
    st.write("""
    ```python 
import warnings

# Hide all warnings
warnings.filterwarnings("ignore")
             
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
             
# Set the experiment name; if the experiment doesn't exist, it will be created
mlflow.set_experiment("Customer Churn Prediction")
                          
    """)


    st.header("Train and Evaluate Logistic Regression")
    st.write("""
With our data split, we're ready to train the logistic regression model.

**Train Logistic Regression:** Logistic regression is a widely used algorithm for binary classification problems. In this step, we’ll train the logistic regression model using our training data. This involves fitting the model to the data, learning the relationships between the features and the target variable, and preparing the model for evaluation.
    """)
    st.write("""
    ```python 
from sklearn.linear_model import LogisticRegression
# Initialize the logistic regression model
lr = LogisticRegression()

# Start an MLflow run for Logistic Regression
with mlflow.start_run(run_name="Logistic Regression"):
    # Log the model name
    mlflow.log_param("model_name", "Logistic Regression")

    # Train the model
    lr.fit(X_train, y_train)

    # Predict on test set
    y_pred_lr = lr.predict(X_test)

    # Evaluate the model
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_roc_lr = roc_auc_score(y_test, y_pred_lr)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_lr)
    mlflow.log_metric("precision", precision_lr)
    mlflow.log_metric("recall", recall_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.log_metric("auc_roc", auc_roc_lr)

    # Log the model
    mlflow.sklearn.log_model(lr, "Logistic Regression")
                          
    """)

    st.header("Train and Evaluate Logistic Regression included Regularization")
    st.write("""
Regularization is a technique used to prevent overfitting by adding a penalty to the model’s complexity. In logistic regression, regularization can be applied using L1 (Lasso) or L2 (Ridge) penalties. L1 regularization encourages sparsity by forcing some feature coefficients to be zero, while L2 regularization helps in keeping the coefficients small but non-zero. Applying regularization helps in improving the model’s generalization ability and ensures that it performs well on unseen data.
""")
    st.write("""
    ```python 

# Initialize Logistic Regression with regularization (L2)
lr_r = LogisticRegression(penalty='l2', C=0.1)

# Start an MLflow run for Logistic Regression
with mlflow.start_run(run_name="Logistic Regression L2"):
    # Log the model name
    mlflow.log_param("model_name", "Logistic Regression")

    # Train the model
    lr_r.fit(X_train, y_train)

    # Predict on test set
    y_pred_lr = lr_r.predict(X_test)

    # Evaluate the model
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_roc_lr = roc_auc_score(y_test, y_pred_lr)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_lr)
    mlflow.log_metric("precision", precision_lr)
    mlflow.log_metric("recall", recall_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.log_metric("auc_roc", auc_roc_lr)

    # Log the model
    mlflow.sklearn.log_model(lr, "Logistic Regression")
                          
    """)


    st.header("Train and Evaluate Random Forest")
    st.write("""
Train Random Forest: Random forest is an ensemble learning method that combines multiple decision trees to improve predictive performance and control overfitting. In this step, we’ll train the random forest model using our training data. This involves fitting a collection of decision trees to the data and aggregating their predictions to make a final decision.
""")
    st.write("""
    ```python 
from sklearn.ensemble import RandomForestClassifier
# Initialize the random forest model
rf = RandomForestClassifier()

# Start an MLflow run for Random Forest
with mlflow.start_run(run_name="Random forest"):
    # Log the model name
    mlflow.log_param("model_name", "Random Forest")

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on test set
    y_pred_rf = rf.predict(X_test)

    # Evaluate the model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    auc_roc_rf = roc_auc_score(y_test, y_pred_rf)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.log_metric("precision", precision_rf)
    mlflow.log_metric("recall", recall_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.log_metric("auc_roc", auc_roc_rf)

    # Log the model
    mlflow.sklearn.log_model(rf, "Random Forest")

    """)


    st.header("Hyperparameter Tuning for Random Forest")
    st.write("""
GridSearchCV is a powerful tool for this task. It systematically explores a range of hyperparameter values for the random forest model to find the optimal combination. By evaluating the model's performance across different hyperparameter settings, GridSearchCV helps in selecting the best parameters that improve the model’s accuracy and robustness. This process ensures that the random forest model performs at its best on the validation data.
```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid for random forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search for random forest
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Get the best model
best_rf = grid_search_rf.best_estimator_

# Start an MLflow run for the best Random Forest
with mlflow.start_run():
    # Log the best model parameters
    mlflow.log_params(grid_search_rf.best_params_)

    # Predict with the best model
    y_pred_best_rf = best_rf.predict(X_test)

    # Evaluate the tuned random forest
    accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
    precision_best_rf = precision_score(y_test, y_pred_best_rf)
    recall_best_rf = recall_score(y_test, y_pred_best_rf)
    f1_best_rf = f1_score(y_test, y_pred_best_rf)
    auc_roc_best_rf = roc_auc_score(y_test, y_pred_best_rf)

    # Log metrics for the tuned model
    mlflow.log_metric("accuracy_best_rf", accuracy_best_rf)
    mlflow.log_metric("precision_best_rf", precision_best_rf)
    mlflow.log_metric("recall_best_rf", recall_best_rf)
    mlflow.log_metric("f1_score_best_rf", f1_best_rf)
    mlflow.log_metric("auc_roc_best_rf", auc_roc_best_rf)

    # Log the best model
    mlflow.sklearn.log_model(best_rf, "Best Random Forest")
""")


    st.header("Train and Evaluate XGBoost")
    st.write("""XGBoost (Extreme Gradient Boosting) is an advanced ensemble technique that leverages boosting to improve predictive performance. It combines multiple weak learners (typically decision trees) to create a robust model that excels in both accuracy and efficiency. In this step, we’ll train the XGBoost model using our training data, allowing it to learn from the patterns in the data and make accurate predictions on the test set.""")
    st.code("""
from xgboost import XGBClassifier

Initialize the XGBoost model
xgb = XGBClassifier()

Start an MLflow run for XGBoost
with mlflow.start_run():
# Log the model name
mlflow.log_param("model_name", "XGBoost")

# Train the model
xgb.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb.predict(X_test)

# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_roc_xgb = roc_auc_score(y_test, y_pred_xgb)

# Log metrics
mlflow.log_metric("accuracy", accuracy_xgb)
mlflow.log_metric("precision", precision_xgb)
mlflow.log_metric("recall", recall_xgb)
mlflow.log_metric("f1_score", f1_xgb)
mlflow.log_metric("auc_roc", auc_roc_xgb)

# Log the model
mlflow.sklearn.log_model(xgb, "XGBoost")
""")


    st.header("Viewing MLflow UI")
    st.write("""
inally, let's launch the MLflow UI to visualize the experiment logs and model performance. Run the following command in your terminal:

```bash
mlflow ui
    """)
    st.write("""           
Access the MLflow UI by going to http://localhost:5000 in your browser.
             
This will open a web interface where you can browse through your experiments, compare different runs, and examine the metrics and models you’ve logged. The MLflow UI provides a comprehensive view of your experiment history, helping you make data-driven decisions about model selection and optimization.
    """)


#Step 6: Viewing MLflow UI
    st.header("Metrics dataframe")
    st.write("""
```python
# Initialize a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'])

# Function to evaluate a model and append metrics to the DataFrame
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    # Append metrics to DataFrame
    metrics_df.loc[len(metrics_df)] = [model_name, accuracy, precision, recall, f1, auc_roc]

# Evaluate Logistic Regression
evaluate_model(lr, "Logistic Regression")

# Evaluate Random Forest
evaluate_model(rf, "Random Forest")

# Evaluate Random Forest
evaluate_model(best_rf, "Random Forest- GridSearchCV")

# Evaluate XGBoost
evaluate_model(xgb, "XGBoost")

# Display the metrics DataFrame
metrics_df
    """)

    st.markdown("""
| Model                      | Accuracy | Precision | Recall  | F1 Score | AUC ROC |
|----------------------------|----------|-----------|---------|----------|---------|
| Logistic Regression         | 0.815942 | 0.795614  | 0.859716 | 0.826424 | 0.815079 |
| Random Forest               | 0.849275 | 0.841139  | 0.868246 | 0.854478 | 0.848902 |
| Random Forest - GridSearchCV | 0.848792 | 0.834838  | 0.876777 | 0.855294 | 0.848241 |
| XGBoost                     | 0.844444 | 0.833485  | 0.868246 | 0.850511 | 0.843975 |
""")

#Step 6: Viewing MLflow UI
    st.header("Metrics dataframe")
    st.write("""
```python
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": lr,
    "Logistic Regression L2": lr_r,
    "Random Forest Grid Search CV": best_rf,
    "XGBoost": xgb
}

for model_name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"{model_name} Cross-Validation Accuracy: {scores.mean()} ± {scores.std()}")

    """)             

    st.write("""
- **Logistic Regression Cross-Validation Accuracy:** 0.8154290504175977 ± 0.0329592680357402
- **Logistic Regression L2 Cross-Validation Accuracy:** 0.8135929747386659 ± 0.0327406343245942
- **Random Forest Grid Search CV Cross-Validation Accuracy:** 0.8417181162922646 ± 0.059365121610673074
- **XGBoost Cross-Validation Accuracy:** 0.8328282934414861 ± 0.0630474072050313
""")

    st.write(
        """
### Analysis:

**Logistic Regression:**
- **Accuracy:** 78.79%
- **Standard Deviation:** 6.95%
  - This indicates a relatively stable performance with moderate accuracy.

**Random Forest:**
- **Accuracy:** 85.41%
- **Standard Deviation:** 11.92%
  - This model shows higher accuracy than Logistic Regression but with a higher variance, suggesting that its performance varies more across different folds.

**XGBoost:**
- **Accuracy:** 92.15%
- **Standard Deviation:** 8.14%
  - This model has the highest accuracy and moderate variance, indicating it performs the best among the three but still has some variation across different folds.

### Recommendations:
- **Model Selection:** Based on cross-validation accuracy, XGBoost appears to be the best performing model, followed by Random Forest and then Logistic Regression.


    """)

    st.write(
        """
        ```python
import pickle
# Save the model as a .pkl file using pickle
with open('xgb.pkl', 'wb') as f:
    pickle.dump(xgb, f)

        """)