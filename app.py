import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import base64

# -----------------------------
# 1. Page Setup
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("💰 Loan Default Prediction System")
st.write("Predict the likelihood of borrowers defaulting on loans using trained Machine Learning and Deep Learning models.")

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
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Model Performance", "🧠 Predict Default Risk", "📂 Sample Data"])

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

    col1, col2 = st.columns(2)
    with col1:
        st.image("models/roc_curves.png", caption="ROC Curves Comparison", use_container_width=True)
    with col2:
        st.image("models/feature_importance.png", caption="Top 15 Feature Importances (Random Forest)", use_container_width=True)

# -----------------------------
# 5. Model Performance Section
# -----------------------------
elif menu == "📊 Model Performance":
    st.header("📊 Model Performance Comparison")
    try:
        df = pd.read_csv("models/model_performance_comparison.csv")
        st.dataframe(df, use_container_width=True)

        st.bar_chart(df.set_index('Model')["Accuracy"])
    except FileNotFoundError:
        st.error("Performance comparison file not found. Please ensure 'model_performance_comparison.csv' exists in the 'models/' folder.")

# -----------------------------
# 6. Prediction Section
# -----------------------------
elif menu == "🧠 Predict Default Risk":
    st.header("🧠 Make a Prediction")

    uploaded_file = st.file_uploader("📂 Upload Borrower Data (CSV)", type=["csv"])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(user_df.head())

        try:
            # Scale input data
            X_scaled = scaler.transform(user_df)

            # Predictions
            st.subheader("🔍 Model Predictions")
            preds = {
                "Logistic Regression": lr_model.predict_proba(X_scaled)[:,1],
                "Random Forest": rf_model.predict_proba(X_scaled)[:,1],
                "MLP": mlp_model.predict_proba(X_scaled)[:,1],
                "Deep Neural Network": dnn_model.predict(X_scaled).ravel()
            }

            results_df = pd.DataFrame(preds)
            results_df["Average Probability"] = results_df.mean(axis=1)
            results_df["Predicted Default"] = (results_df["Average Probability"] >= 0.5).astype(int)

            st.write("### 🧾 Prediction Results:")
            st.dataframe(results_df)

            st.write("### 📈 Default Risk Overview:")
            st.bar_chart(results_df[["Average Probability"]])

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    else:
        st.info("Upload a CSV file containing borrower data to generate predictions.")

# -----------------------------
# 7. Sample Data Section
# -----------------------------
elif menu == "📂 Sample Data":
    st.header("📂 Sample Borrower Data Format")
    st.markdown("""
    Below is a sample format for the borrower data you should upload.  
    Each row represents a borrower's profile.
    """)

    sample_data = pd.DataFrame({
        "Age": [35, 42, 29],
        "Income": [55000, 42000, 68000],
        "LoanAmount": [15000, 10000, 20000],
        "Loan_Term": [36, 24, 48],
        "Credit_Score": [720, 650, 780],
        "NumCreditLines": [4, 2, 5],
        "HasMortgage": [1, 0, 1],
        "HasCoSigner": [0, 1, 0],
        "Employment_Length": [5, 10, 3],
        "Existing_Loan_Balance": [2000, 0, 5000]
    })

    st.dataframe(sample_data)

    # Download button for CSV
    csv = sample_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_input.csv">📥 Download Sample CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed for Loan Default Prediction Project — using Machine Learning & Deep Learning models.")
