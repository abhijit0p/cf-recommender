import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "interactions.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cf_ranker.pkl")

class RankedCF:
    def __init__(self):
        self.df = pd.read_json(DATA_PATH)
        # Create Pivot: Users x Items
        self.matrix = self.df.pivot_table(index='user', columns='item', values='rating').fillna(0)
        # Calculate Item-Item Similarity
        self.item_sim = cosine_similarity(self.matrix.T)
        self.sim_df = pd.DataFrame(self.item_sim, index=self.matrix.columns, columns=self.matrix.columns)
        self.ranker = joblib.load(MODEL_PATH)
        
        # Precompute global popularity
        self.popularity = self.df['item'].value_counts() / len(self.df)

    def recommend(self, item_name, k=3, min_support=3, sim_floor=0.20):
        if item_name not in self.sim_df.columns: return None
        
        # 1. Get ALL similarities
        all_candidates = self.sim_df[item_name].drop(item_name)
        
        # 2. SIMILARITY VETO: Only high overlaps are candidates
        valid_candidates = all_candidates[all_candidates > sim_floor]
        
        if valid_candidates.empty:
            return []

        scored_results = []
        for cand_name, sim_score in valid_candidates.items():
            pop = self.popularity.get(cand_name, 0)
            # 3. SUPPORT VETO: Require at least 5 shared users
            # This kills the 'random bridge' between API Gateway and Cookers
            support = ((self.matrix[item_name] > 0) & (self.matrix[cand_name] > 0)).sum()
            
            if support < min_support:
                continue

            features = np.array([[sim_score, pop, support]])
            final_prob = self.ranker.predict_proba(features)[0][1]
            scored_results.append((cand_name, final_prob))

        return sorted(scored_results, key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    engine = RankedCF()
    seeds = ["iPhone 17", "API Gateway", "Prestige Cooker"]
    
    print("\n" + "="*50)
    print("PRODUCTION-GRADE HYBRID-CF TEST")
    print("="*50)
    
    for seed in seeds:
        print(f"\nTarget: [{seed}]")
        recs = engine.recommend(seed)
        if recs:
            for name, score in recs:
                print(f"  -> {name:20} | Ranker Score: {score:.4f}")
        else:
            print("  -> (No patterns met the high-confidence threshold)")