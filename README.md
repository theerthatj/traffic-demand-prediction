# Traffic Demand Prediction System


#  Overview

The **Traffic Demand Prediction System** is an end-to-end Machine Learning project that predicts hourly traffic demand using historical traffic, weather, road infrastructure, and temporal information.

The project compares multiple regression algorithms, applies ensemble learning to improve prediction accuracy, and presents the results through an interactive **Streamlit** web application.

In addition to traffic demand prediction, the project also includes advanced prediction modules for congestion estimation, accident risk prediction, and route recommendation.

---

##  Model Performance Dashboard

Compare the performance of multiple regression models using:

- R² Score
- RMSE
- MAE
- MAPE

Supported models:

- Random Forest
- XGBoost
- LightGBM
- Stacking Regressor
- Weighted Ensemble

---

##  Advanced Prediction Modules

- Congestion Level Estimation
- Accident Risk Prediction
- Intelligent Route Recommendation

---

#  Tech Stack

| Category | Technologies |
|-----------|-------------|
| Programming Language | Python |
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Ensemble Models | Random Forest, XGBoost, LightGBM |
| Visualization | Matplotlib |
| Model Serialization | Joblib |

---

#  Project Structure

```text
traffic-demand-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── traffic_dataset.csv
│   ├── traffic_featured.csv
│   └── model_performance.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── stacked_model.pkl
│   ├── feature_columns.pkl
│   ├── accident_risk_classifier.pk1
│   ├── congestion_classifier.pk1
│   ├── geohash_encoder.pkl
│   ├── road_encoder.pkl
│   ├── weather_encoder.pkl
│   └── landmark_encoder.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_ensemble_learning.ipynb
│   └── 06_advanced_predictions.ipynb
│
├── reports/
│   ├── data_dictionary.md
│   └── report.md
│
├── src/
│   └── ensemble_model.py
│
├── visualizations/
│   ├── traffic_distribution.png
│   ├── traffic_by_hour.png
│   ├── traffic_by_road_type.png
│   ├── weather_impact_analysis.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── ...
│
└── .streamlit/
    └── config.toml
```

---

#  Dataset

The dataset contains **120,000 synthetic traffic observations** with temporal, weather, infrastructure, and traffic-related features.

### Main Features

- Hour
- Day of Week
- Month
- Geohash Location
- Road Type
- Number of Lanes
- Traffic Signals
- Large Vehicle Count
- Temperature
- Humidity
- Rainfall
- Weather Conditions
- Nearby Landmarks

### Target Variable

- **Traffic Demand (vehicles/hour)**

---

#  Feature Engineering

The following engineered features were created:

- Peak Hour Flag
- Weekend Flag
- Rush Hour Indicator
- Signal Density
- Weather Impact Score
- Traffic Density Score
- Large Vehicle Count Normalized
- Traffic Density Index
- Accident Risk Score

These engineered features improved model performance by incorporating traffic domain knowledge.

---

#  Evaluation Metrics

Models were evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

The comparison dashboard displays all evaluation metrics in a single table for easy comparison.

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/traffic-demand-prediction.git
```

Move into the project directory:

```bash
cd traffic-demand-prediction
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your default browser.

---

#  Application Modules

##  Prediction

Users can input:

- Geohash Zone
- Road Type
- Weather Condition
- Date & Time
- Number of Lanes
- Large Vehicle Count

Outputs:

- Predicted Traffic Demand
- Congestion Level
- Peak Traffic Alert
- 24-Hour Forecast

---

##  EDA Dashboard

Displays saved visualizations including:

- Traffic Distribution
- Hourly Traffic Analysis
- Road Type Analysis
- Weather Impact
- Correlation Matrix
- Feature Importance

---

##  Model Performance

Displays:

- Model comparison table
- Performance metrics
- Best-performing model
- Comparative charts

---


#  Documentation

A detailed project report is available in:

```
-  `reports/report.md` – Complete project report
-  `reports/data_dictionary.md` – Dataset documentation
```

#  Future Improvements

- Real-time traffic API integration
- Live weather API integration
- GPS-based traffic visualization
- Deep Learning (LSTM/Transformer)
- Real-time IoT traffic monitoring
- Cloud deployment
- Mobile application
- Automatic model retraining

---

