from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('/home/benny/drug_project/models/xgb_model.pkl')  # Replace 'benny' with your username
scaler = joblib.load('/home/benny/drug_project/models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [data['age'], data['gender'], data['income'], data['education'], data['mental_health']]
    scaled = scaler.transform([features])
    prob = model.predict_proba(scaled)[0][1]
    risk = "High" if prob > 0.5 else "Low"
    return jsonify({'risk_probability': float(prob), 'risk_level': risk, 'recommendation': 'Seek counseling if High.'})

if __name__ == '__main__':
    app.run()
