import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class CFRecommender:
    def __init__(self, data_path):
        self.df = pd.read_json(data_path)
        # Create the Pivot Table: Rows = Users, Columns = Items
        self.matrix = self.df.pivot_table(index='user', columns='item', values='rating').fillna(0)

    def get_user_based(self, target_user, k=2):
        """Find users similar to target_user and see what they liked."""
        if target_user not in self.matrix.index:
            return "User not found"
            
        sim_matrix = cosine_similarity(self.matrix)
        sim_df = pd.DataFrame(sim_matrix, index=self.matrix.index, columns=self.matrix.index)
        
        # Get similar users
        similar_users = sim_df[target_user].sort_values(ascending=False)[1:k+1].index
        
        # Items these users liked that target_user hasn't bought
        user_items = self.matrix.loc[target_user]
        recs = self.matrix.loc[similar_users].mean().sort_values(ascending=False)
        return recs[user_items == 0].head(3)

    def get_item_based(self, target_item, k=2):
        """Amazon Style: Find items frequently bought with target_item."""
        if target_item not in self.matrix.columns:
            return "Item not found"
            
        # Transpose matrix to get Item-to-Item similarity
        item_sim_matrix = cosine_similarity(self.matrix.T)
        item_sim_df = pd.DataFrame(item_sim_matrix, index=self.matrix.columns, columns=self.matrix.columns)
        
        # Get items similar to the target item based on user behavior
        return item_sim_df[target_item].sort_values(ascending=False)[1:k+1]

# Execution
recommender = CFRecommender("interactions.json")

print("--- User-Based (Personalization) ---")
print("Target u2 (bought iPhone17) might also like:")
print(recommender.get_user_based("u2"))

print("\n--- Item-Based (Product Overlap / Amazon Style) ---")
print("People who viewed 'iPhone17' also viewed:")
print(recommender.get_item_based("iPhone17"))