import pandas as pd

print("--- LOADING SPAIN GRID ENERGY FILE ---")
energy_df = pd.read_csv("energy_dataset.csv")
print(f"Energy Data Shape: {energy_df.shape[0]} rows and {energy_df.shape[1]} columns.")
print("Energy Data Columns:", energy_df.columns.tolist()[:5], "...and more.\n")

print("--- LOADING SPAIN WEATHER FEATURES FILE ---")
weather_df = pd.read_csv("weather_features.csv")
print(f"Weather Data Shape: {weather_df.shape[0]} rows and {weather_df.shape[1]} columns.")
print("Weather Data Columns:", weather_df.columns.tolist()[:5], "...and more.")