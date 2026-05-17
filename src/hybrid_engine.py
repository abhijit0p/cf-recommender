import numpy as np
from cf_recommender import RankedCF
# Assuming your previous Semantic Engine is in 'semantic_engine.py'
# from semantic_engine import SemanticSearch 

class HybridEngine:
    def __init__(self):
        self.cf_engine = RankedCF()
        # Mocking the Semantic Engine response for this demo
        self.semantic_scores = {
            ("iPhone 17", "Magsafe Charger"): 0.95,
            ("iPhone 17", "Tiramisu"): 0.05,
            ("API Gateway", "Cloud Run"): 0.98,
            ("API Gateway", "Prestige Cooker"): 0.02
        }

    def get_hybrid_recs(self, item_name, k=3):
        # 1. Get Behavioral Candidates from the CF Engine
        cf_results = self.cf_engine.recommend(item_name, k=10)
        if not cf_results: return []

        hybrid_results = []
        for cand_name, cf_score in cf_results:
            # 2. Get Semantic Score (Content Relevance)
            # In production, this would call your Vector DB / SentenceTransformer
            sem_score = self.semantic_scores.get((item_name, cand_name), 0.5)
            
            # 3. The Weighted Hybrid Formula
            # We give 70% weight to behavior (the crowd) and 30% to content (logic)
            final_score = (cf_score * 0.7) + (sem_score * 0.3)
            
            hybrid_results.append((cand_name, final_score, cf_score, sem_score))

        # 4. Re-rank based on the new hybrid score
        return sorted(hybrid_results, key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    engine = HybridEngine()
    for seed in ["iPhone 17", "API Gateway"]:
        print(f"\nHybrid Results for [{seed}]:")
        for name, total, cf, sem in engine.get_hybrid_recs(seed):
            print(f"  -> {name:18} | Hybrid: {total:.4f} (CF: {cf:.2f}, Sem: {sem:.2f})")