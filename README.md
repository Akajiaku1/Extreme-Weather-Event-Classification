**Extreme Weather Event Classification Project**

https://img.freepik.com/free-vector/weather-concept-illustration_114360-1234.jpg

This project classifies extreme weather events using synthetic data and machine learning techniques. It demonstrates how to generate realistic weather data and build a classification model to identify different types of extreme weather conditions.
Features

The model uses five key weather parameters as input features:

    Temperature (°C)

    Humidity (%)

    Precipitation (mm)

    Wind Speed (km/h)

    Soil Moisture (%)

Event Classes

The classifier identifies six types of weather events:
Class	Event Type	Description
0	Normal Weather	Typical weather conditions
1	Heatwave	Prolonged period of excessively hot weather
2	Coldwave	Prolonged period of excessively cold weather
3	Flood	Overflow of water submerging land
4	Drought	Prolonged period of low precipitation
5	Wildfire	Uncontrolled fire in vegetation areas
Project Structure
text

Extreme-Weather-Event-Classification/
├── data/
│   └── extreme_weather_dataset.csv  # Generated synthetic dataset
├── images/
│   ├── feature_importance.png      # Feature importance visualization
│   └── confusion_matrix.png        # Model performance visualization
├── extreme_weather_classification.py  # Main classification script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation

Installation and Usage
Prerequisites

    Python 3.8+

    pip package manager

Installation

    Clone the repository:

bash

git clone https://github.com/Akajiaku1/Extreme-Weather-Event-Classification.git
cd Extreme-Weather-Event-Classification

    Install dependencies:

bash

pip install -r requirements.txt

Running the Project

Execute the main script:
bash

python extreme_weather_classification.py

This will:

    Generate a synthetic weather dataset (1000 days)

    Save the dataset to data/extreme_weather_dataset.csv

    Train a Random Forest classifier

    Evaluate model performance

    Generate visualizations in the images/ directory

Sample Output
Classification Report
text

                precision    recall  f1-score   support

Normal Weather       0.95      0.99      0.97       170
     Heatwave       0.94      0.84      0.89        31
     Coldwave       0.90      0.86      0.88        29
        Flood       0.97      0.90      0.93        30
      Drought       0.90      0.90      0.90        29
     Wildfire       0.94      0.94      0.94        31

    accuracy                           0.94       320
   macro avg       0.93      0.90      0.92       320
weighted avg       0.94      0.94      0.94       320

Visualizations

https://images/feature_importance.png
https://images/confusion_matrix.png
Key Features

    Synthetic Data Generation: Creates realistic weather data with temporal correlations

    Physical Constraints: Ensures parameters stay within realistic ranges

    Event Signatures: Each event type has characteristic parameter modifications

    Balanced Classification: Uses class weighting to handle imbalanced data

    Visual Analytics: Generates feature importance and confusion matrix plots

Customization

You can modify the following parameters in the script:

    DAYS: Number of days to generate in the dataset

    EVENT_TYPES: Add or modify event types

    Model hyperparameters (n_estimators, test_size, etc.)

    Event characteristics in the generate_events function

Author

Ugochukwu Charles Akajiaku

    GitHub: Akajiaku1

    Project Repository: https://github.com/Akajiaku1/Extreme-Weather-Event-Classification

License

This project is licensed under the MIT License - see the LICENSE file for details.
Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.
