import joblib
import numpy as np

# Load saved models and scaler
lr = joblib.load('/home/benny/drug_project/models/lr_model.pkl')
rf = joblib.load('/home/benny/drug_project/models/rf_model.pkl')
xgb = joblib.load('/home/benny/drug_project/models/xgb_model.pkl')
scaler = joblib.load('/home/benny/drug_project/models/scaler.pkl')

def predict_risk(model, input_features):
    scaled = scaler.transform([input_features])
    prob = model.predict_proba(scaled)[0][1]
    risk = "High" if prob > 0.5 else "Low"
    return f"Risk Probability: {prob:.2f} ({risk}) - Recommendation: Seek counseling if High."

# Example prediction (age=30, gender=1, income=50000, education=3, mental_health=0)
sample = [30, 1, 50000, 3, 0]
print("Logistic Regression:", predict_risk(lr, sample))
print("Random Forest:", predict_risk(rf, sample))
print("XGBoost:", predict_risk(xgb, sample))
