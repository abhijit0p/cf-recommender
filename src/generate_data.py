import json
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "interactions.json")

def generate_5k_messy_data():
    clusters = {
        "GCP_Dev": ["API Gateway", "Cloud Run", "Firestore", "BigQuery", "MacBook Pro"],
        "Italian_Dining": ["Spaghetti Carbonara", "Margherita Pizza", "Chianti Wine", "Tiramisu", "Garlic Bread"],
        "Apple_Ecosystem": ["iPhone 17", "AirPods", "Magsafe Charger", "iPad Pro", "Apple Pencil"],
        "Home_Smart": ["Prestige Cooker", "Induction Cooktop", "Smart Fan", "Air Fryer", "Mixing Bowls"],
        "Fitness": ["Yoga Mat", "Dumbbells", "HIIT Guide", "Resistance Bands", "Kettlebell"],
        "Travel": ["Backpacking Bag", "Travel Adapter", "Hiking Boots", "Passport Holder"]
    }
    
    chaos_items = ["Fevicol", "Undergarments", "Dish Soap", "Batteries", "Door Mat", "USB Cable", "Light Bulb", "Laundry Detergent"]
    all_items = [i for sub in clusters.values() for i in sub] + chaos_items
    
    interactions = []
    
    # 1. 500 Unique Users x 10 items each = 5,000 baseline
    for u_idx in range(500):
        user_id = f"user_{u_idx}"
        primary_cat = random.choice(list(clusters.keys()))
        
        for _ in range(10):
            # 40% Persona Consistency / 60% Pure Chaos
            if random.random() < 0.40:
                item = random.choice(clusters[primary_cat])
            else:
                item = random.choice(all_items)
            
            interactions.append({"user": user_id, "item": item, "rating": 1})

    # 2. Add 'Power Users' (Heavy noise markers)
    for p_idx in range(10):
        power_user = f"power_user_{p_idx}"
        for _ in range(50):
            interactions.append({"user": power_user, "item": random.choice(all_items), "rating": 1})

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'w') as f:
        json.dump(interactions, f, indent=2)
        
    print(f"Dataset Scaled! Generated {len(interactions)} interactions.")

if __name__ == "__main__":
    generate_5k_messy_data()