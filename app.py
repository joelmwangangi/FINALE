import streamlit as st
import pandas as pd
import base64

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
        st.info("📊 ROC Curves comparison will be displayed here.")
    with col2:
        st.info("📈 Feature importance chart will be displayed here.")

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
# 5. Prediction Section (Interactive UI)
# -----------------------------
elif menu == "🧠 Predict Default Risk":
    st.header("🧠 Predict Default Risk")

    st.subheader("Enter Borrower Data Manually")
    with st.form("borrower_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
        loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=60, value=36)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
        has_mortgage = st.selectbox("Has Mortgage?", options=["No", "Yes"])
        has_cosigner = st.selectbox("Has Co-Signer?", options=["No", "Yes"])
        employment_length = st.number_input("Employment Length (years)", min_value=0, value=5)
        existing_loan_balance = st.number_input("Existing Loan Balance ($)", min_value=0, value=0)

        submitted = st.form_submit_button("Predict Default Risk")

    if submitted:
        # Convert categorical fields to numeric placeholders
        has_mortgage_num = 1 if has_mortgage == "Yes" else 0
        has_cosigner_num = 1 if has_cosigner == "Yes" else 0

        # Create dataframe with one row (like a model input)
        input_df = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan_amount],
            "Loan_Term": [loan_term],
            "Credit_Score": [credit_score],
            "NumCreditLines": [num_credit_lines],
            "HasMortgage": [has_mortgage_num],
            "HasCoSigner": [has_cosigner_num],
            "Employment_Length": [employment_length],
            "Existing_Loan_Balance": [existing_loan_balance]
        })

        st.write("### Borrower Data Preview")
        st.dataframe(input_df)

        # Placeholder predictions
        st.write("### 🧾 Predicted Default Probabilities (Placeholder)")
        results_df = pd.DataFrame({
            "Logistic Regression": [0.45],
            "Random Forest": [0.50],
            "MLP": [0.48],
            "Deep Neural Network": [0.52],
            "Average Probability": [0.49],
            "Predicted Default": [0]  # 0 = No Default, 1 = Default
        })
        st.dataframe(results_df)
        st.write("### 📈 Default Risk Overview")
        st.bar_chart([0.49])  # placeholder for average probability chart

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
st.caption("Developed for Loan Default Prediction Project — UI-only version with interactive borrower input.")
