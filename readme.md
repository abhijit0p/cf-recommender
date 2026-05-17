# Production-Grade Hybrid Collaborative Filtering Recommender

A modular, high-threshold recommendation engine that combines **Behavioral Collaborative Filtering (CF)** with **Semantic Content Relevance**, stabilized by a machine learning-based veto ranker to filter out real-world data noise and "chaos" interactions.

## 🚀 Overview

Real-world interaction logs are messy. This engine is built to handle highly noisy datasets by implementing a multi-stage recommendation pipeline:
1. **Behavioral Discovery:** Computes Item-Item Cosine Similarity profiles based on user interaction matrices.
2. **Veto Layers:** Applies strict **Similarity Floor** and **Support Count** filters to eliminate spurious or accidental co-occurrences.
3. **ML Re-Ranking:** Evaluates remaining candidates through a trained Logistic Regression ranker (`cf_ranker.pkl`) assessing similarity, global item popularity, and co-purchase support.
4. **Hybrid Blending:** Blends the behavioral confidence scores (70% weight) with semantic/content alignment vectors (30% weight) to produce structurally sound, context-aware recommendations.

---

## 📂 Project Structure

```text
cf-recommender/
│
├── data/
│   └── interactions.json       # Generated synthetic messy interaction logs
│
├── models/
│   └── cf_ranker.pkl           # Trained Logistic Regression vetting model
│
├── src/
│   ├── generate_data.py        # Scaled data generator (5k+ clustered/chaos rows)
│   ├── train_ranker.py         # Sci-Kit Learn training script for the veto ranker
│   ├── cf_engine.py            # Baseline User/Item-based CF reference
│   ├── cf_recommender.py       # Production-grade Ranked Collaborative Filtering
│   └── hybrid_engine.py        # Main engine blending behavioral & semantic scores
│
├── .gitignore                  # Prevents environment and temporary file leaks
└── README.md                   # System documentation