import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Features: [Similarity, Global_Popularity_Ratio, Support_Count]
# [Similarity, Global_Popularity_Ratio, Support_Count]
X = [
    [0.60, 0.05, 5],  # Good Sim, Niche, Support 5 -> 1 (Strong)
    [0.40, 0.10, 3],  # Lower Sim, Support 3 -> 1 (Acceptable Signal)
    [0.20, 0.50, 2],  # Low Sim, Popular item, Support 2 -> 0 (Noise)
    [0.80, 0.90, 2],  # High Sim but it's a 'Chaos' item -> 0
]
y = [1, 1, 0, 0]

model = LogisticRegression()
model.fit(X, y)
joblib.dump(model, "models/cf_ranker.pkl")
print("High-Threshold Ranker trained.")