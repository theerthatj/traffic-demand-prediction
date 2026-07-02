# Data Dictionary

## Traffic Demand Prediction System

This document describes all features used in the **Traffic Demand Prediction System** dataset, including their data types, expected ranges, and purpose.

---

# Dataset Overview

| Property | Value |
|----------|-------|
| Dataset Name | Traffic Demand Dataset |
| Records | 120,000 |
| Features | 25 |
| Target Variable | `traffic_demand` |
| Data Type | Mixed (Numerical & Categorical) |

---

# Feature Descriptions

| Feature | Data Type | Range / Categories | Description |
|----------|-----------|-------------------|-------------|
| **timestamp** | Datetime | YYYY-MM-DD HH:MM:SS | Date and time of the traffic observation. |
| **hour** | Integer | 0-23 | Hour of the day extracted from the timestamp. |
| **day_of_week** | Integer | 0-6 | Day of the week (0 = Monday, 6 = Sunday). |
| **geohash_location** | Categorical | gc00 - gc49 | Encoded traffic observation zone identifier. |
| **road_type** | Categorical | Highway, Arterial, Local, Collector, Expressway | Category of road where traffic is observed. |
| **num_lanes** | Integer | 1-6 | Number of available traffic lanes. |
| **traffic_signals** | Integer | 0-10 | Number of traffic signals near the observation point. |
| **large_vehicles_count** | Integer | 0-50 | Number of buses, trucks, and other heavy vehicles. |
| **temperature** | Float | Approximately -10°C to 45°C | Ambient air temperature. |
| **humidity** | Float | 0-100% | Relative humidity percentage. |
| **rainfall** | Float | 0-100 mm | Rainfall recorded during the observation period. |
| **weather_conditions** | Categorical | Clear, Cloudy, Rainy, Stormy, Foggy, Snowy | Weather condition affecting traffic flow. |
| **nearby_landmarks** | Categorical | Airport, Hospital, School, Mall, Stadium, Park, Train Station | Prominent nearby landmark influencing traffic. |
| **traffic_demand** | Float | ~100-2500 vehicles/hour | **Target variable** representing hourly traffic volume. |
| **month** | Integer | 1-12 | Month extracted from the timestamp. |
| **event_Concert** | Boolean | True / False | Indicates whether a concert is occurring nearby. |
| **event_Conference** | Boolean | True / False | Indicates whether a conference is occurring nearby. |
| **event_Festival** | Boolean | True / False | Indicates whether a festival is occurring nearby. |
| **event_Sports_Event** | Boolean | True / False | Indicates whether a sporting event is occurring nearby. |
| **peak_hour_flag** | Integer | 0 or 1 | Indicates whether the observation falls within peak traffic hours. |
| **weekend_flag** | Integer | 0 or 1 | Indicates whether the observation occurs on a weekend. |
| **rush_hour_indicator** | Integer | 0 or 1 | Binary indicator representing rush-hour traffic conditions. |
| **signal_density** | Float | >0 | Ratio of traffic signals to the number of lanes. |
| **weather_impact_score** | Float | Continuous | Composite score representing the combined impact of weather variables on traffic. |
| **traffic_density_score** | Float | 0-1 | Normalized heavy-vehicle density score derived from large vehicle count. |

---

# Target Variable

## traffic_demand

| Property | Value |
|----------|-------|
| Data Type | Float |
| Unit | Vehicles per Hour |
| Approximate Range | 100 - 2500 |
| Objective | Regression |

The **traffic_demand** feature represents the estimated number of vehicles passing through a traffic observation point in one hour and serves as the primary prediction target for the machine learning models.

---

# Engineered Features

The following features were generated during feature engineering to improve predictive performance.

---

## Peak Hour Flag

**Type:** Integer (0/1)

**Formula**

```text
Peak Hour Flag =
1 if Hour ∈ {7,8,9,17,18,19}
Else 0
```

**Purpose**

Captures morning and evening rush-hour traffic.

---

## Weekend Flag

**Type:** Integer (0/1)

**Formula**

```text
Weekend Flag =
1 if Day of Week ≥ 5
Else 0
```

**Purpose**

Differentiates weekday and weekend traffic behavior.

---

## Rush Hour Indicator

**Type:** Integer (0/1)

Represents peak traffic periods.

---

## Signal Density

**Type:** Float

**Formula**

```text
Signal Density = Traffic Signals/Number of Lanes
```

**Purpose**

Measures traffic control intensity relative to road capacity.

---

## Weather Impact Score

**Type:** Float

**Formula**

```text
Weather Impact Score =
0.5 × Rainfall
+
0.3 × Humidity
+
0.2 × Temperature
```

**Purpose**

Represents the combined influence of weather conditions on traffic flow.

---

## Traffic Density Score

**Type:** Float

**Formula**

```text
Traffic Density Score = Large Vehicle Count/50
```

**Purpose**

Normalizes heavy vehicle concentration to a 0-1 scale.

---

# Categorical Features

| Feature | Categories |
|----------|------------|
| Road Type | Highway, Arterial, Collector, Local, Expressway |
| Weather Conditions | Clear, Cloudy, Rainy, Stormy, Foggy, Snowy |
| Nearby Landmarks | Airport, Hospital, School, Shopping Mall, Stadium, Park, Train Station |
| Geohash Location | gc00-gc49 |

---

# Data Preprocessing Summary

The following preprocessing operations were applied before model training:

- Removal of duplicate records
- Handling of missing values
- Datetime conversion
- Temporal feature extraction
- Label encoding of categorical variables
- Feature engineering
- Train-test split (80:20)
- Consistent feature ordering for all trained models

---

# Notes

- The dataset used in this project is **synthetically generated** for educational and research purposes.
- Geohash identifiers represent simulated traffic zones rather than real geographic coordinates.
- All engineered features were created to improve the predictive capability of the machine learning models while maintaining interpretability.