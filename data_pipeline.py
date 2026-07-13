import pandas as pd

def process_spain_data():
    print("Step 1: Ingesting raw files into memory...")
    energy_df = pd.read_csv("energy_dataset.csv")
    weather_df = pd.read_csv("weather_features.csv")
    
    print("Step 2: Filtering weather profiles to Madrid...")
    madrid_weather = weather_df[weather_df['city_name'] == 'Madrid'].copy()
    
    print("Step 3: Standardizing timestamps...")
    energy_df['time'] = pd.to_datetime(energy_df['time'], utc=True)
    madrid_weather['time'] = pd.to_datetime(madrid_weather['dt_iso'], utc=True)
    
    print("Step 4: Stitching datasets together (The Merge)...")
    merged_df = pd.merge(energy_df, madrid_weather, on='time', how='inner')
    merged_df = merged_df.set_index('time')
    
    print("Step 5: Renaming columns and creating AI clues...")
    merged_df = merged_df.rename(columns={
        'total load actual': 'Load',
        'price actual': 'Price',
        'generation solar': 'Solar_Gen',
        'temp': 'Temperature',
        'wind_speed': 'Wind_Speed'
    })
    
    final_cols = ['Load', 'Price', 'Solar_Gen', 'Temperature', 'Wind_Speed']
    cleaned_df = merged_df[final_cols].dropna()
    
    cleaned_df['hour'] = cleaned_df.index.hour
    cleaned_df['day_of_week'] = cleaned_df.index.dayofweek
    cleaned_df['month'] = cleaned_df.index.month
    
    cleaned_df['load_lag_24h'] = cleaned_df['Load'].shift(24)
    cleaned_df = cleaned_df.dropna()
    
    print("Pipeline Complete! Cleaned Dataset generated.")
    return cleaned_df

if __name__ == "__main__":
    result_data = process_spain_data()
    print(f"\nFinal Master Data Specs: {result_data.shape[0]} rows aligned across {result_data.shape[1]} metrics!")
    print("\nSample Preview:")
    print(result_data[['Load', 'Price', 'Solar_Gen', 'Temperature', 'hour']].head(3))