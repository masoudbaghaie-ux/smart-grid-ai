import pandas as pd
from xgboost import XGBRegressor
import pickle
from data_pipeline import process_spain_data

def train_ai_systems():
    # 1. Pull the clean, unified data we built in Step 2
    df = process_spain_data()
    
    # 2. Define the Clues (Features) for each model
    # Demand needs historical reference hooks (lag); Solar just needs basic time-space positioning
    features_demand = ['hour', 'day_of_week', 'month', 'load_lag_24h']
    features_supply = ['hour', 'month', 'Temperature']
    
    # 3. Create a Chronological Train/Test Split
    # We take all historical rows except the final 720 hours (the last 30 days) to use for learning.
    # We save those last 30 days to test the model's true accuracy later.
    train_df = df.iloc[:-720]
    
    # 4. Initialize the Two Independent XGBoost AI Engines
    # 'n_estimators=100' means build 100 sequential decision trees to refine answers.
    # 'learning_rate=0.05' scales down each tree's step size to avoid overshooting patterns.
    model_demand = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model_supply = XGBRegressor(n_estimators=100, learning_rate=0.05)
    
    # 5. The Training Phase (The .fit() command forces the AI to study)
    print("\n[AI Train] Studying consumer demand consumption patterns...")
    model_demand.fit(train_df[features_demand], train_df['Load'])
    
    print("[AI Train] Studying solar radiation and generation trends...")
    model_supply.fit(train_df[features_supply], train_df['Solar_Gen'])
    
    # 6. Saving the Finished Brains (Serialization)
    # This freezes the calculated mathematical patterns into static files
    # so our web dashboard can load them instantly without needing to re-train.
    with open('model_demand.pkl', 'wb') as f:
        pickle.dump(model_demand, f)
    with open('model_supply.pkl', 'wb') as f:
        pickle.dump(model_supply, f)
        
    print("\nAI Training Phase Finalized! Two model files successfully compiled to your workspace.")

if __name__ == "__main__":
    train_ai_systems()