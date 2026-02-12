import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# -----------------------------
# 1. Page Setup
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("💰 Loan Default Prediction System")
st.write("This app predicts the likelihood of a borrower defaulting on a loan using trained ML and Deep Learning models.")

# -----------------------------
# 2. Load Models & Assets
# -----------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.joblib")
    lr_model = joblib.load("models/logistic_regression_model.joblib")
    rf_model = joblib.load("models/random_forest_model.joblib")
    mlp_model = joblib.load("models/mlp_model.joblib")
    dnn_model = tf.keras.models.load_model("models/deep_learning_model.h5")
    return scaler, lr_model, rf_model, mlp_model, dnn_model

scaler, lr_model, rf_model, mlp_model, dnn_model = load_models()

# -----------------------------
# 3. Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Model Performance", "🧠 Predict Default Risk"])

# -----------------------------
# 4. Home Section
# -----------------------------
if menu == "🏠 Home":
    st.header("Overview")
    st.markdown("""
    This project builds and compares multiple models to predict **loan default risk**.
    - **Models Used:** Logistic Regression, Random Forest, Multi-Layer Perceptron, Deep Neural Network  
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC  
    - **Outputs:** ROC Curves, Feature Importance, Model Performance Summary
    """)

    st.image("models/roc_curves.png", caption="ROC Curves Comparison", use_container_width=True)
    st.image("models/feature_importance.png", caption="Top 15 Feature Importances (Random Forest)", use_container_width=True)

# -----------------------------
# 5. Model Performance Section
# -----------------------------
elif menu == "📊 Model Performance":
    st.header("Model Performance Comparison")
    df = pd.read_csv("models/model_performance_comparison.csv")
    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index('Model')["Accuracy"])

# -----------------------------
# 6. Prediction Section
# -----------------------------
elif menu == "🧠 Predict Default Risk":
    st.header("Make a Prediction")

    uploaded_file = st.file_uploader("Upload Borrower Data (CSV)", type=["csv"])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(user_df.head())

        # Preprocess input
        X_scaled = scaler.transform(user_df)

        # Predict with all models
        st.subheader("Model Predictions")
        preds = {
            "Logistic Regression": lr_model.predict_proba(X_scaled)[:,1],
            "Random Forest": rf_model.predict_proba(X_scaled)[:,1],
            "MLP": mlp_model.predict_proba(X_scaled)[:,1],
            "Deep Neural Network": dnn_model.predict(X_scaled).ravel()
        }

        results_df = pd.DataFrame(preds)
        results_df["Average Probability"] = results_df.mean(axis=1)
        results_df["Predicted Default"] = (results_df["Average Probability"] >= 0.5).astype(int)

        st.write("### Prediction Results:")
        st.dataframe(results_df)

        # Summary
        st.write("### Default Risk Summary:")
        st.bar_chart(results_df[["Average Probability"]])

    else:
        st.info("Please upload a CSV file containing borrower data to get predictions.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed for Loan Default Prediction Project — using Deep Learning and Machine Learning models.")
