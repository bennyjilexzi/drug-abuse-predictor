import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000
data = {
    'age': np.random.randint(18, 65, n),
    'gender': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    'income': np.random.randint(20000, 150000, n),
    'education': np.random.randint(1, 5, n),
    'mental_health': np.random.choice([0, 1], n, p=[0.7, 0.3])
}
risk_score = (
    0.3 * (data['mental_health'] == 1).astype(int) +
    0.2 * (data['age'] < 30).astype(int) +
    0.1 * (data['income'] < 50000).astype(int) +
    np.random.normal(0, 0.1, n)
)
data['drug_use'] = (risk_score > 0.4).astype(int)
df = pd.DataFrame(data)
df.to_csv('/home/benny/drug_project/data/nsduh_sample.csv', index=False)  
df.to_csv('/home/benny/drug_project/data/nsduh_sample.csv', index=False)
print(f"Synthetic dataset created: {df.shape[0]} rows.")
print(df.head())
print(df['drug_use'].value_counts())

