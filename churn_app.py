# ============================================================
# PROJECT 1: Churn Prediction - Streamlit Dashboard
# Run: streamlit run churn_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction Dashboard")

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("churn_model.json")
    return model

model = load_model()

st.sidebar.header("Enter Customer Details")
tenure        = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charge = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
total_charges  = st.sidebar.number_input("Total Charges ($)", 0, 10000, 800)
contract       = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet       = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support   = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
support_map  = {"No": 0, "No internet service": 1, "Yes": 2}
payment_map  = {
    "Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
    "Electronic check": 2, "Mailed check": 3
}

input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charge,
    "TotalCharges": total_charges,
    "Contract": contract_map[contract],
    "InternetService": internet_map[internet],
    "TechSupport": support_map[tech_support],
    "PaymentMethod": payment_map[payment_method],
    # Add remaining features with defaults
    "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
    "PhoneService": 1, "MultipleLines": 0,
    "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
    "StreamingTV": 0, "StreamingMovies": 0, "PaperlessBilling": 1,
    "gender": 0
}])

col1, col2 = st.columns(2)
prob = model.predict_proba(input_data)[0][1]
pred = model.predict(input_data)[0]

with col1:
    st.subheader("Churn Probability")
    color = "red" if prob > 0.5 else "green"
    st.markdown(f"<h1 style='color:{color}'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
    st.write("**Prediction:**", "Will Churn" if pred == 1 else "Will Stay")

with col2:
    st.subheader("Risk Level")
    if prob > 0.7:
        st.error("HIGH RISK - Immediate action needed")
    elif prob > 0.4:
        st.warning("MEDIUM RISK - Monitor this customer")
    else:
        st.success("LOW RISK - Customer is likely to stay")

st.subheader("Feature Importance (SHAP)")
explainer = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(input_data)
fig, ax = plt.subplots()
shap.waterfall_plot(shap.Explanation(
    values=shap_vals[0],
    base_values=explainer.expected_value,
    feature_names=input_data.columns.tolist()
), show=False)
st.pyplot(fig)
