# extreme_weather_classification.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Constants
DAYS = 1000
EVENT_TYPES = {
    0: 'Normal Weather',
    1: 'Heatwave',
    2: 'Coldwave',
    3: 'Flood',
    4: 'Drought',
    5: 'Wildfire'
}

# Generate base features with temporal correlation
def generate_base_features(n_days):
    # Initialize with normal weather baseline
    temp = np.random.normal(25, 5, n_days)
    humidity = np.random.normal(50, 10, n_days)
    precipitation = np.abs(np.random.normal(5, 3, n_days))
    wind_speed = np.abs(np.random.normal(15, 5, n_days))
    soil_moisture = np.random.normal(50, 10, n_days)
    
    # Add temporal correlation (today similar to yesterday)
    for i in range(1, n_days):
        temp[i] = 0.7 * temp[i-1] + 0.3 * temp[i]
        humidity[i] = 0.7 * humidity[i-1] + 0.3 * humidity[i]
        precipitation[i] = 0.6 * precipitation[i-1] + 0.4 * precipitation[i]
        wind_speed[i] = 0.6 * wind_speed[i-1] + 0.4 * wind_speed[i]
        soil_moisture[i] = 0.8 * soil_moisture[i-1] + 0.2 * soil_moisture[i]
    
    return temp, humidity, precipitation, wind_speed, soil_moisture

# Generate extreme events
def generate_events(features):
    temp, humidity, precip, wind, soil = features
    n_days = len(temp)
    labels = np.zeros(n_days, dtype=int)  # Default: Normal weather
    
    # Randomly select event days (15% of days)
    event_days = np.random.choice(n_days, size=int(n_days*0.15), replace=False)
    
    for day in event_days:
        event_type = np.random.choice([1,2,3,4,5])  # Exclude normal weather
        
        if event_type == 1:  # Heatwave
            temp[day] += np.random.uniform(8, 15)
            humidity[day] *= np.random.uniform(0.4, 0.7)
            precip[day] = np.random.uniform(0, 2)
            soil[day] *= np.random.uniform(0.4, 0.6)
            
        elif event_type == 2:  # Coldwave
            temp[day] -= np.random.uniform(10, 20)
            humidity[day] *= np.random.uniform(0.8, 1.2)
            precip[day] = np.random.uniform(0, 5)
            wind[day] += np.random.uniform(5, 15)
            
        elif event_type == 3:  # Flood
            precip[day] += np.random.uniform(30, 100)
            humidity[day] += np.random.uniform(20, 40)
            soil[day] += np.random.uniform(30, 50)
            wind[day] += np.random.uniform(5, 20)
            
        elif event_type == 4:  # Drought
            precip[day] = np.random.uniform(0, 1)
            humidity[day] *= np.random.uniform(0.3, 0.6)
            soil[day] *= np.random.uniform(0.2, 0.4)
            temp[day] += np.random.uniform(3, 8)
            
        elif event_type == 5:  # Wildfire
            temp[day] += np.random.uniform(5, 12)
            humidity[day] *= np.random.uniform(0.3, 0.6)
            wind[day] += np.random.uniform(10, 30)
            soil[day] *= np.random.uniform(0.3, 0.5)
            
        labels[day] = event_type
        
    return labels

# Generate dataset
def generate_dataset(n_days):
    features = generate_base_features(n_days)
    labels = generate_events(features)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': features[0],
        'humidity': features[1],
        'precipitation': features[2],
        'wind_speed': features[3],
        'soil_moisture': features[4],
        'event': labels
    })
    
    # Clip values to physical limits
    df['temperature'] = df['temperature'].clip(-20, 50)
    df['humidity'] = df['humidity'].clip(0, 100)
    df['precipitation'] = df['precipitation'].clip(0, 200)
    df['wind_speed'] = df['wind_speed'].clip(0, 100)
    df['soil_moisture'] = df['soil_moisture'].clip(0, 100)
    
    return df

# Generate and save dataset
weather_df = generate_dataset(DAYS)
weather_df.to_csv('extreme_weather_dataset.csv', index=False)

# Train classification model
def train_model(df):
    # Prepare data
    X = df.drop('event', axis=1)
    y = df['event']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EVENT_TYPES.values()))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model

# Main execution
if __name__ == "__main__":
    print(f"Generated dataset with {DAYS} days of weather data")
    weather_df = generate_dataset(DAYS)
    print("Dataset sample:")
    print(weather_df.head())
    
    # Save dataset
    weather_df.to_csv('extreme_weather_dataset.csv', index=False)
    print("Dataset saved to 'extreme_weather_dataset.csv'")
    
    # Train and evaluate model
    print("\nTraining classification model...")
    model = train_model(weather_df)
    print("Model training complete!")