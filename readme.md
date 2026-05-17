# Production-Grade Hybrid Collaborative Filtering Recommender

A modular recommendation engine combining:

- Behavioral Collaborative Filtering (CF)
- Semantic Content Similarity
- ML-based ranking filters

The system is designed to handle noisy real-world interaction data by filtering accidental co-occurrences and blending behavioral confidence with semantic relevance.

---

# ✨ Features

- Item-Item Collaborative Filtering using cosine similarity
- Noise suppression using similarity thresholds and support counts
- Logistic Regression-based veto/ranking model
- Hybrid recommendation blending:
  
```text
Final Score =
    (CF Score × 0.7)
  + (Semantic Score × 0.3)
```

- Modular architecture for experimentation and scaling

---

# 📂 Project Structure

```text
cf-recommender/
│
├── data/
│   └── interactions.json
│
├── models/
│   └── cf_ranker.pkl
│
├── src/
│   ├── generate_data.py
│   ├── train_ranker.py
│   ├── cf_engine.py
│   ├── cf_recommender.py
│   └── hybrid_engine.py
│
├── .gitignore
└── README.md
```

---

# ⚙️ Setup

## Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install numpy pandas scikit-learn joblib
```

---

# 🚀 Execution Flow

## Step 1 — Generate Mock Interaction Data

Creates clustered and noisy interaction logs.

```bash
python src/generate_data.py
```

---

## Step 2 — Train the Ranker

Trains the Logistic Regression veto model.

```bash
python src/train_ranker.py
```

---

## Step 3 — Run Collaborative Filtering Engine

```bash
python src/cf_recommender.py
```

Example output:

```text
Target: [iPhone 17]
  -> AirPods
  -> Apple Pencil
  -> Magsafe Charger
```

---

## Step 4 — Run Hybrid Recommendation Engine

```bash
python src/hybrid_engine.py
```

Example output:

```text
Hybrid Results for [iPhone 17]:

  -> Magsafe Charger
     Hybrid: 0.9850

  -> AirPods
     Hybrid: 0.8500
```

---

# 🧠 Recommendation Pipeline

## 1. Behavioral Candidate Discovery

Builds Item-Item similarity using cosine similarity over interaction matrices.

---

## 2. Noise Filtering

Recommendations are filtered using:
- Similarity thresholds
- Minimum support counts

This removes weak or accidental correlations.

---

## 3. ML Re-Ranking

A Logistic Regression model evaluates:
- Similarity strength
- Shared interaction support
- Item popularity

---

## 4. Hybrid Blending

Final recommendations combine:
- Behavioral relevance
- Semantic similarity

This improves contextual recommendation quality.

---

# 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---

# 📌 Future Improvements

- FAISS / ANN retrieval
- Transformer embeddings
- Online learning
- Time-decay ranking
- XGBoost rankers

---

# 📜 License

MIT
