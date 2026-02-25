import streamlit as st
import joblib
import numpy as np
import os

# Page Configuration
st.set_page_config(
    page_title="Drug Abuse Risk Assessment Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models - Fixed paths for cloud deployment
@st.cache_resource
def load_models():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load models using relative paths
    model = joblib.load(os.path.join(project_dir, 'models', 'xgb_model.pkl'))
    scaler = joblib.load(os.path.join(project_dir, 'models', 'scaler.pkl'))
    return model, scaler

model, scaler = load_models()

# Title and Header
st.title("Drug Abuse Risk Assessment Tool")
st.markdown("---")
st.info("**Purpose:** This tool uses machine learning to assess an individual's risk of drug abuse based on demographic and behavioral factors. It is designed for **prevention and early intervention** purposes only and should not be used as a diagnostic tool. For concerns, please contact a healthcare professional or SAMHSA at 1-800-662-HELP.")

# Create Two Columns for Input
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Personal Information")
    
    # Age: Research shows drug abuse often starts around 12-25
    age = st.number_input("Age", min_value=12, max_value=100, value=25, help="Average age for drug abuse initiation is typically 12-25 years")
    
    # Gender: Male/Female instead of 0/1
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender_map = {"Male": 0, "Female": 1}
    gender_value = gender_map[gender]
    
    # Currency and Income
    st.header("Financial Information")
    currency = st.selectbox("Currency", ["Naira (NGN)", "USD ($)", "GBP (£)", "EUR (€)"])
    currency_multiplier = {"Naira (NGN)": 1, "USD ($)": 1500, "GBP (£)": 1900, "EUR (€)": 1650}
    income_input = st.number_input(f"Annual Income ({currency})", min_value=0, value=500000, step=10000)
    income = income_input * currency_multiplier[currency]

with col2:
    st.header("Educational Background")
    
    # Education: Actual levels instead of 1-4
    education = st.selectbox("Highest Education Level", [
        "Primary School",
        "Secondary School",
        "Some College/University",
        "Bachelor's Degree",
        "Master's Degree",
        "Doctorate/PhD"
    ])
    education_map = {
        "Primary School": 1,
        "Secondary School": 2,
        "Some College/University": 3,
        "Bachelor's Degree": 4,
        "Master's Degree": 5,
        "Doctorate/PhD": 6
    }
    education_value = education_map[education]
    
    st.header("Mental Health")
    
    # Mental Health: Yes/No instead of 0/1
    mental_health = st.radio("History of Mental Health Issues?", ["No", "Yes"], horizontal=True)
    mental_health_value = 0 if mental_health == "No" else 1

# Prediction Button
st.markdown("---")
if st.button("Assess Risk", use_container_width=True):
    # Prepare features
    features = [age, gender_value, income, education_value, mental_health_value]
    scaled = scaler.transform([features])
    prob = model.predict_proba(scaled)[0][1]
    risk = "High" if prob > 0.5 else "Moderate" if prob > 0.3 else "Low"
    
    st.markdown("---")
    st.header("Risk Assessment Results")
    
    # Display Results
    if risk == "High":
        st.warning(f"**High Risk Detected**\n\nRisk Probability: {float(prob):.1%}\n\nRecommendation: Please seek professional counseling immediately. Contact SAMHSA at 1-800-662-HELP or speak with a healthcare provider.")
    elif risk == "Moderate":
        st.info(f"**Moderate Risk Detected**\n\nRisk Probability: {float(prob):.1%}\n\nRecommendation: Consider speaking with a counselor or mental health professional for guidance and support.")
    else:
        st.success(f"**Low Risk Detected**\n\nRisk Probability: {float(prob):.1%}\n\nRecommendation: Continue maintaining healthy habits. Stay informed about drug abuse prevention.")
    
    # Progress Bar
    st.progress(float(prob))
    st.caption(f"Risk Level: {float(prob):.1%}")

# Footer
st.markdown("---")
st.caption("Drug Abuse Risk Assessment Tool | Powered by Machine Learning (XGBoost Algorithm) | Data Source: National Survey on Drug Use and Health (NSDUH) 2021")
