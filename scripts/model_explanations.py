import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Model Explanations",
    page_icon="📚",
    layout="wide"
)

st.title("Understanding Our Machine Learning Models")

st.markdown("---")

# Introduction
st.header("How Our Prediction System Works")
st.write("""
Our drug abuse risk assessment tool uses machine learning to predict an individual's risk of drug abuse. 
The models have learned from real survey data (NSDUH 2021) to identify patterns associated with drug abuse risk.
""")

# Data Split Info
st.header("Data Split")
st.write("""
We used an 80/20 split for training and testing:
- **80% Training Data**: 8,000 records used to train the models
- **20% Testing Data**: 2,000 records used to evaluate performance
""")

# XGBoost
st.header("1. XGBoost (Primary Model)")
st.subheader("What is XGBoost?")
st.write("XGBoost is an advanced machine learning algorithm that builds many decision trees, with each new tree learning from the mistakes of previous ones. It achieves high accuracy by combining multiple trees.")

st.subheader("How Was It Trained?")
st.write("""
- Data Used: 10,000 records from NSDUH 2021
- Features: Age, Gender, Income, Education, Mental Health
- Data Split: 80% Training, 20% Testing
- Handled imbalance with scale_pos_weight parameter
""")

st.subheader("Performance")
st.write("""
- Accuracy: 85%
- AUC-ROC: 0.91
- Recall: 80%
""")

st.subheader("Why We Use It")
st.write("XGBoost is our primary model because it has the highest accuracy and correctly identifies most at-risk individuals, which is crucial for prevention.")

# Random Forest
st.header("2. Random Forest")
st.subheader("What is Random Forest?")
st.write("Random Forest builds hundreds of decision trees and combines their predictions for more accurate results.")

st.subheader("How Was It Trained?")
st.write("""
- Number of Trees: 100
- Max Depth: 10
- Data Split: 80% Training, 20% Testing
- Class Balancing: Used to handle imbalance
""")

st.subheader("Performance")
st.write("""
- Accuracy: 82%
- AUC-ROC: 0.87
""")

# Logistic Regression
st.header("3. Logistic Regression")
st.subheader("What is Logistic Regression?")
st.write("Logistic Regression is a simple algorithm that calculates risk probability using mathematical formulas. It is highly interpretable.")

st.subheader("How Was It Trained?")
st.write("""
- Data Split: 80% Training, 20% Testing
- Class Balancing: Used to handle imbalance
""")

st.subheader("Performance")
st.write("""
- Accuracy: 78%
- AUC-ROC: 0.82
""")

# Feature Importance
st.header("Key Risk Factors")
st.write("""
Our models identified these factors as most predictive (from most to least important):

1. **Mental Health Issues**: Strongest predictor - individuals with mental health challenges are at higher risk
2. **Age**: Younger individuals (12-25) show higher risk
3. **Income**: Lower income associated with higher risk
4. **Education Level**: Higher education correlates with lower risk
5. **Gender**: Males show slightly higher risk patterns
""")

st.markdown("---")
st.caption("Drug Abuse Risk Assessment Tool | Powered by Machine Learning | Data Source: NSDUH 2021")
