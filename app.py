import streamlit as st
import pandas as pd
import base64
import os

# -----------------------------
# 1. Page Setup
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("💰 Loan Default Prediction System")
st.write(
    "Predict the likelihood of borrowers defaulting on loans using Machine Learning and Deep Learning models."
)

# -----------------------------
# 2. Sidebar Navigation
# -----------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Model Performance", "🧠 Predict Default Risk", "📂 Sample Data"]
)

# -----------------------------
# 3. Home Section
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
        st.image("placeholder_roc.png", caption="ROC Curves Comparison", use_container_width=True)
    with col2:
        st.image("placeholder_feature_importance.png", caption="Top Feature Importances", use_container_width=True)

# -----------------------------
# 4. Model Performance Section
# -----------------------------
elif menu == "📊 Model Performance":
    st.header("📊 Model Performance Comparison")

    # Placeholder dataframe
    data = {
        "Model": ["Logistic Regression", "Random Forest", "MLP", "Deep Neural Network"],
        "Accuracy": [0.85, 0.88, 0.86, 0.90],
        "Precision": [0.80, 0.87, 0.82, 0.89],
        "Recall": [0.78, 0.85, 0.81, 0.88],
        "F1": [0.79, 0.86, 0.81, 0.88]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index('Model')["Accuracy"])

# -----------------------------
# 5. Prediction Section (UI Only)
# -----------------------------
elif menu == "🧠 Predict Default Risk":
    st.header("🧠 Make a Prediction")
    uploaded_file = st.file_uploader("📂 Upload Borrower Data (CSV)", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview:")
        st.dataframe(user_df.head())

        # Placeholder for prediction results
        st.write("### 🧾 Prediction Results (Placeholder)")
        st.dataframe(pd.DataFrame({
            "Logistic Regression": [0.0] * len(user_df),
            "Random Forest": [0.0] * len(user_df),
            "MLP": [0.0] * len(user_df),
            "Deep Neural Network": [0.0] * len(user_df),
            "Average Probability": [0.0] * len(user_df),
            "Predicted Default": [0] * len(user_df)
        }))

        st.write("### 📈 Default Risk Overview (Placeholder)")
        st.bar_chart([0] * len(user_df))
    else:
        st.info("Upload a CSV file containing borrower data to see predictions here.")

# -----------------------------
# 6. Sample Data Section
# -----------------------------
elif menu == "📂 Sample Data":
    st.header("📂 Sample Borrower Data Format")
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
st.caption("Developed for Loan Default Prediction Project — UI-only version.")
