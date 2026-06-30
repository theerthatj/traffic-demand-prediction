# Traffic Demand Prediction System

## Project Report

**Project Title:** Traffic Demand Prediction System using Machine Learning<br>
**Project Type:** Machine Learning 


#  Project Objective & Scope

## Objectives

The primary objectives of this project are:

* Predict hourly traffic demand using historical traffic information.
* Compare multiple machine learning regression algorithms.
* Improve prediction accuracy using ensemble learning.
* Build an interactive Streamlit dashboard.
* Assist intelligent transportation systems through data-driven prediction.
* Demonstrate advanced predictive modules such as congestion estimation, accident risk prediction, and route recommendation.

## Scope

The project focuses on:

* Traffic demand prediction
* Traffic analytics
* Machine learning model comparison
* Ensemble learning
* Interactive visualization
* Decision support for transportation planning

The project uses a synthetic dataset for experimentation and educational purposes.

---

#  Dataset Description

## Dataset Overview

The dataset contains historical traffic information collected from multiple synthetic geohash zones and includes temporal, environmental, and infrastructure-related attributes.

### Major Features

| Category  | Features                                            |
| --------- | --------------------------------------------------- |
| Temporal  | Hour, Day of Week, Month                            |
| Location  | Geohash Zone                                        |
| Road      | Road Type, Number of Lanes                          |
| Traffic   | Large Vehicle Count, Traffic Signals                |
| Weather   | Temperature, Humidity, Rainfall, Weather Conditions |
| Landmarks | Nearby Landmarks                                    |
| Target    | Traffic Demand                                      |

### Engineered Features

* Peak Hour Flag
* Weekend Flag
* Rush Hour Indicator
* Signal Density
* Weather Impact Score
* Traffic Density Score
* Large Vehicle Count Normalized
* Traffic Density Index
* Accident Risk Score


#  Data Preprocessing

The following preprocessing steps were performed before model training:

* Removal of duplicate records.
* Handling of missing values.
* Conversion of timestamps into datetime format.
* Extraction of temporal features (hour, day of week, month).
* Label Encoding of categorical variables:

  * Geohash Location
  * Road Type
  * Weather Conditions
  * Nearby Landmarks
* Separation of regression and classification targets.
* Train-test split using an 80:20 ratio.
* Consistent feature ordering for all trained models.

---

#  Feature Engineering

Feature engineering significantly improved model performance by incorporating domain-specific knowledge.

## Peak Hour Flag

**Formula**

```
Peak Hour Flag = 1 if hour ∈ {7,8,9,17,18,19}
Else 0
```

**Purpose**

Captures rush-hour traffic patterns.

---

## Weekend Flag

**Formula**

```
Weekend Flag = 1 if DayOfWeek ≥ 5
Else 0
```

**Purpose**

Differentiates weekday and weekend traffic behavior.

---

## Rush Hour Indicator

```
Rush Hour Indicator = Peak Hour Flag
```

Used to explicitly represent rush-hour conditions.

---

## Signal Density

```
Signal Density = Traffic Signals / Number of Lanes
```

Represents traffic control intensity.

---

## Weather Impact Score

```
Weather Impact Score =
0.5 × Rainfall
+
0.3 × Humidity
+
0.2 × Temperature
```

Represents combined environmental influence on traffic.

---

## Traffic Density Score

```
Traffic Density Score =
Large Vehicles
---------------
50
```

Measures relative heavy-vehicle concentration.

---

## Large Vehicle Count Normalized

```
Large Vehicle Count Normalized =
Large Vehicles
---------------
Maximum Vehicles
```

Normalizes heavy-vehicle count.

---

## Traffic Density Index

```
Traffic Density Index =
Large Vehicle Count Normalized
×
Signal Density
```

Combines infrastructure and heavy traffic.

---

## Accident Risk Score

```
Accident Risk Score =
Traffic Density Index
×
(Weather Impact Score + 1)
```

Designed for accident risk estimation.

---

#  Model Training

Four regression models were trained.

## Random Forest Regressor

**Key Hyperparameters**

* n_estimators
* max_depth
* random_state

---

## XGBoost Regressor

**Key Hyperparameters**

* learning_rate
* n_estimators
* max_depth
* subsample

---

## LightGBM Regressor

**Key Hyperparameters**

* learning_rate
* num_leaves
* n_estimators
* random_state

---

## Stacking Regressor

Base learners:

* Random Forest
* XGBoost
* LightGBM

Meta learner:

* Linear Regression

---

## Training Strategy

* 80% Training
* 20% Testing
* Consistent feature ordering
* Performance evaluated using multiple regression metrics

---

#  Results & Evaluation

The following evaluation metrics were used:

* R² Score
* RMSE
* MAE
* MAPE

| Model             |         R² |       RMSE |        MAE |       MAPE |
| ----------------- | ---------: | ---------: | ---------: | ---------: |
| Random Forest     |  0.9207    |    177.6   |   115.47   |   16.06    |
| XGBoost           |  0.9482    |    143.6   |   95.26    |   13.02    |
| LightGBM          |  0.9482    |    143.58  |   95.04    |   13.0     |
| Weighted Ensemble |  0.9486    |    142.92  |   94.64    |   12.91    |
| Stacking Ensemble |  0.9482    |    143.53  |   95.24    |   13.02    |


### Observations

* Ensemble learning improved prediction stability.
* Gradient boosting models achieved high predictive performance.
* The stacking ensemble provided robust overall performance across all metrics.

---

#  Ensemble Strategy

Instead of relying on a single model, predictions were combined using a weighted ensemble approach.

The ensemble integrates predictions from:

* Random Forest
* XGBoost
* LightGBM

Each model contributes according to predefined weights determined during experimentation.

**Advantages**

* Reduced prediction variance
* Improved robustness
* Better generalization
* Increased overall prediction accuracy

---

#  Advanced Prediction Modules

## Congestion Level Estimation

Predicted traffic demand is categorized into:

* Low
* Medium
* High

This provides an intuitive interpretation of predicted traffic conditions.

---

## Accident Risk Prediction

A machine learning classifier estimates accident risk using engineered traffic, weather, and road-related features.

---

## Route Recommendation System

A route recommendation module evaluates candidate routes using the trained ensemble regression model. Each candidate route is scored based on predicted traffic demand, and the least congested alternatives are recommended.

---

#  Streamlit Application

An interactive Streamlit web application was developed to provide a user-friendly interface for the prediction system.

## Modules

### Traffic Demand Prediction

Users can input:

* Location
* Road Type
* Weather
* Date
* Time
* Number of Lanes
* Large Vehicle Count

Outputs include:

* Predicted Traffic Demand
* Congestion Level
* Peak Traffic Alert
* 24-Hour Forecast

---

### Exploratory Data Analysis Dashboard

Displays:

* Traffic Distribution
* Traffic by Road Type
* Weather Impact Analysis
* Correlation Heatmap
* Feature Importance
* Actual vs Predicted Plots
* Other saved visualizations

---

### Model Performance Dashboard

Displays:

* Model comparison table
* Best-performing model
* R² comparison
* RMSE comparison
* MAE comparison
* MAPE comparison

---

#  Conclusion

This project demonstrates the successful application of ensemble machine learning techniques for traffic demand prediction. Through extensive preprocessing, feature engineering, and model comparison, the system effectively predicts hourly traffic demand using multiple influencing factors. The interactive Streamlit dashboard makes the predictions accessible to end users while providing analytical insights through visualizations and model evaluation. The inclusion of advanced prediction modules such as congestion estimation, accident risk prediction, and route recommendation further enhances the practical relevance of the project. Overall, the proposed system provides a scalable framework for intelligent transportation analytics and demonstrates the effectiveness of ensemble learning for real-world traffic prediction problems.

---

#  Future Scope

The project can be further enhanced through several improvements:

* Integration with real-time traffic APIs.
* Use of actual GPS coordinates instead of synthetic geohash identifiers.
* Deployment using cloud platforms such as AWS, Azure, or GCP.
* Deep learning approaches using LSTM or Transformer-based models for time-series forecasting.
* Real-time traffic monitoring using IoT sensors.
* Live congestion heatmaps using mapping services.
* Multi-city traffic prediction.
* Automatic model retraining using streaming traffic data.
* Mobile application integration.
* Intelligent traffic signal optimization based on predicted demand.

---

