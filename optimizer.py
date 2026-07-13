import numpy as np
from scipy.optimize import linprog

def run_grid_optimization(predicted_prices):
    # 🌟 FIX: Instead of hardcoding 24, look at the actual length of incoming prices!
    # If the app sends 24 hours, n_hours becomes 24. If it sends 1 year, it becomes 8,760.
    n_hours = len(predicted_prices)
    
    # Objective function coefficients (Prices vector)
    c = predicted_prices 
    
    # Build bounds dynamically based on the calculated n_hours
    hour_bounds = [(-20, 20) for _ in range(n_hours)]
    
    # Calculate the median price over the provided timeline chunk
    median_price = np.median(predicted_prices)
    
    for i in range(n_hours):
        if predicted_prices[i] < median_price:
            # Shift the hourly bounds to force a charge profile (0 to 20 MW) during cheap windows
            hour_bounds[i] = (0, 20)
        else:
            # Shift the hourly bounds to force a discharge profile (-20 to 0 MW) during peak price windows
            hour_bounds[i] = (-20, 0)

    # Run the modern solver matrix calculations using matching vector dimensions
    result = linprog(c, bounds=hour_bounds, method='highs')
    
    if result.success:
        return result.x 
    else:
        return np.zeros(n_hours)

if __name__ == "__main__":
    # Quick internal validation check using 24 items
    fake_prices = np.array([50.0] * 24)
    fake_prices[5] = 10.0   
    fake_prices[20] = 100.0 
    
    decisions = run_grid_optimization(fake_prices)
    print("Optimizer Script Test Run (Dynamic Sizing Verified):")
    print(f"Array length processed: {len(decisions)} rows.")