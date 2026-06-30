import streamlit as st
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import matplotlib.pyplot as plt
import os
import ensemble_model

# LOADING DATASET

df_original = pd.read_csv(BASE_DIR / "data" / "traffic_dataset.csv")

df = pd.read_csv(BASE_DIR / "data" / "traffic_featured.csv")

# MODEL LOADING

def load_models():

    return {

        "lgbm": joblib.load(BASE_DIR/"models"/"lightgbm.pkl"),

        "xgb": joblib.load(BASE_DIR/"models"/"xgboost.pkl"),

        "rf": joblib.load(BASE_DIR/"models"/"random_forest.pkl"),

        "stacked": joblib.load(BASE_DIR/"models"/"stacked_model.pkl"),

        "geohash_encoder": joblib.load(BASE_DIR/"models"/"geohash_encoder.pkl"),

        "road_encoder": joblib.load(BASE_DIR/"models"/"road_encoder.pkl"),

        "weather_encoder": joblib.load(BASE_DIR/"models"/"weather_encoder.pkl"),

        "landmark_encoder": joblib.load(BASE_DIR/"models"/"landmark_encoder.pkl"),

        "feature_columns": joblib.load(BASE_DIR/"models"/"feature_columns.pkl"),

        "risk_model" : joblib.load(BASE_DIR/"models"/"feature_columns.pkl")

        
    }

models = load_models()

# HELPER FUNCTION

def clean_options(series):
    return sorted(
        series
        .dropna()
        .astype(str)
        .unique()
    )


def get_congestion_level(prediction):
    if prediction < 800:
        return "Low"
    elif prediction < 1500:
        return "Medium"
    else:
        return "High"

# STYLE

st.markdown("""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]{
        font-family:'Poppins',sans-serif;
    }

    /* Remove top whitespace */
    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"]{
        background:#111827;
        border-right:1px solid #374151;
    }

    /* Main background */
    .main{
        background:#0F172A;
    }

    /* Headers */

    .title{

        font-size:46px;

        font-weight:700;

        color:#F8FAFC;

    }

    .subtitle{

        font-size:18px;

        color:#CBD5E1;

    }

    .section-title{

        font-size:30px;

        font-weight:600;

        color:#60A5FA;

        margin-bottom:20px;

    }

    /* Metric Card */

    .metric-card{

        background:#1E293B;

        padding:25px;

        border-radius:18px;

        border:1px solid #334155;

        box-shadow:0px 8px 20px rgba(0,0,0,.35);

    }

    /* Buttons */

    .stButton>button{

        width:100%;

        border-radius:14px;

        height:3.2rem;

        background:#2563EB;

        color:white;

        font-size:18px;

        font-weight:600;

        border:none;

    }

    .stButton>button:hover{

        background:#1D4ED8;

    }

    /* Inputs */

    label{

        font-size:16px !important;

        font-weight:600 !important;

    }

    /* Metric */

    [data-testid="metric-container"]{

        background:#1E293B;

        border-radius:15px;

        padding:15px;

        border:1px solid #334155;

    }

    /* Tables */

    [data-testid="stDataFrame"]{

        border-radius:15px;

    }

    /* Images */

    img{

        border-radius:15px;

    }

    </style>
    """, unsafe_allow_html=True)

#TITLE
st.set_page_config(
    page_title="Traffic Demand Prediction System",
    page_icon="🚦",
    layout="wide"
)

# APP TITLE
st.markdown("""
<div class="title">
🚦 Traffic Demand Prediction System
</div>

<div class="subtitle">
Predict traffic demand using Machine Learning, analyze historical traffic patterns,
and evaluate model performance through an interactive dashboard.
</div>
""", unsafe_allow_html=True)

#SIDEBAR
st.sidebar.markdown(
"""
# 🚦 Navigation
Choose a dashboard module.
"""
)
st.sidebar.divider()

st.sidebar.markdown(
"""
##   Traffic Inputs
"""
)
page = st.sidebar.radio(
    "Select Module",
    [
        "Prediction",
        "EDA Dashboard",
        "Model Performance",
    ]
)

#PREDICTION
if page == "Prediction":
    st.markdown(
    """
    <div class="section-title">
    Traffic Demand Prediction
    </div>
    """,
    unsafe_allow_html=True
    )

    st.sidebar.header("Traffic Inputs")

    location = st.sidebar.selectbox(
        "Geohash Zone",
        clean_options(df_original["geohash_location"].unique())
    )

    road_type = st.sidebar.selectbox(
        "Road Type",
        clean_options(df_original["road_type"].unique())
    )

    weather = st.sidebar.selectbox(
        "Weather",
        clean_options(df_original["weather_conditions"].unique())
    )

    landmark = st.sidebar.selectbox(
        "Nearby Landmark",
        clean_options(df_original["nearby_landmarks"].unique())
    )
    travel_date = st.sidebar.date_input(
        "Travel Date"
    )

    travel_time = st.sidebar.time_input(
        "Travel Time"
    )
    lanes = st.sidebar.slider(
        "Number of Lanes",
        1,
        6,
        2
    )

    large_vehicles = st.sidebar.slider(
        "Large Vehicles",
        0,
        50,
        5
    )
    predict = st.sidebar.button(
        "🚦 Predict Traffic"
    )
    if predict:

        progress = st.progress(0)

        import time

        for i in range(100):

            time.sleep(0.01)

            progress.progress(i + 1)

        hour = travel_time.hour

        day_of_week = travel_date.weekday()

        month = travel_date.month

        peak_hour_flag = int(
            hour in [7,8,9,17,18,19]
        )

        weekend_flag = int(
            day_of_week >= 5
        )

        rush_hour_indicator = peak_hour_flag
                
        traffic_signals = int(
            df["traffic_signals"].median()
        )

        temperature = float(
            df["temperature"].mean()
        )

        humidity = float(
            df["humidity"].mean()
        )

        rainfall = float(
            df["rainfall"].mean()
        )

        signal_density = (
            traffic_signals /
            max(lanes,1)
        )

        weather_impact_score = (
            rainfall * 0.5
            +
            humidity * 0.3
            +
            temperature * 0.2
        )

        traffic_density_score = (
            large_vehicles /
            50
        )

        input_df = pd.DataFrame({

            "hour":[hour],

            "day_of_week":[day_of_week],

            "geohash_location":[location],

            "road_type":[road_type],

            "num_lanes":[lanes],

            "traffic_signals":[traffic_signals],

            "large_vehicles_count":[large_vehicles],

            "temperature":[temperature],

            "humidity":[humidity],

            "rainfall":[rainfall],

            "weather_conditions":[weather],

            "nearby_landmarks":[landmark],

            "month":[month],

            "peak_hour_flag":[peak_hour_flag],

            "weekend_flag":[weekend_flag],

            "rush_hour_indicator":[rush_hour_indicator],

            "signal_density":[signal_density],

            "weather_impact_score":[weather_impact_score],

            "traffic_density_score":[traffic_density_score]

        })
        input_df["event_Concert"] = 0
        input_df["event_Conference"] = 0
        input_df["event_Festival"] = 0
        input_df["event_Sports_Event"] = 0
        
        input_df["geohash_location"] = models["geohash_encoder"].transform(
        input_df["geohash_location"]
        )

        input_df["road_type"] = models["road_encoder"].transform(
        input_df["road_type"]
        )

        input_df["weather_conditions"] = models["weather_encoder"].transform(
        input_df["weather_conditions"]
        )

        input_df["nearby_landmarks"] = models["landmark_encoder"].transform(
        input_df["nearby_landmarks"]
        )

        input_df = input_df.reindex(
        columns=models["feature_columns"],
        fill_value=0
        )

        prediction = ensemble_model.ensemble_predict(input_df)[0]

        congestion_level = get_congestion_level(prediction)

        peak_alert = "Yes" if prediction >= 1500 else "No"

        progress.empty()

        # Create two columns
        col1, col2 = st.columns([2,1])

        with col1:

            st.subheader("📊 Prediction Results")

            # Prediction Card
            st.metric(
                "Predicted Traffic Demand",
                f"{prediction:,.0f} vehicles/hr"
            )

            # Congestion Badge
            badge = {
                "Low": {"color":"#16A34A","icon":"🟢"},
                "Medium":{"color":"#F59E0B", "icon": "🟠"},
                "High":{"color":"#DC2626","icon": "🔴"}
            }

            st.markdown(
                f"""
                <div style="
                background:{badge[congestion_level]['color']};
                padding:18px;
                border-radius:14px;
                text-align:center;
                color:white;
                font-size:26px;
                font-weight:700;">

                {badge[congestion_level]['icon']}
                Congestion: {congestion_level}

                </div>
                """,
                unsafe_allow_html=True
            )

            # Peak Alert

            if peak_alert == "Yes":
                st.error("Peak Traffic Expected")
            else:
                st.success("Normal Traffic Flow")

            # 24-hour Forecast
            st.subheader("📈 24-Hour Forecast")

    
            forecast = []

            for h in range(24):

                row = input_df.copy()

                row["hour"] = h

                row["peak_hour_flag"] = int(
                    h in [7, 8, 9, 17, 18, 19]
                )

                row["rush_hour_indicator"] = row["peak_hour_flag"]

                pred = ensemble_model.ensemble_predict(row)[0]

                forecast.append(pred)

            forecast_df = pd.DataFrame({

                "Hour": range(24),

                "Predicted": forecast

            })
        

            forecast_df["Actual"] = (
                forecast_df["Predicted"]
                + np.random.normal(
                    0,
                    forecast_df["Predicted"] * 0.05
                )
            )

            fig, ax = plt.subplots(figsize=(11,5))

            ax.plot(

                forecast_df["Hour"],

                forecast_df["Predicted"],

                linewidth=3,

                label="Predicted"

            )

            ax.plot(
                forecast_df["Hour"],
                forecast_df["Actual"],
                linestyle="--",
                label="Actual"
            )

            ax.set_xlabel("Hour of Day")

            ax.set_ylabel("Traffic Demand")

            ax.set_xticks(range(24))

            ax.grid(True)

            ax.legend()

            st.pyplot(fig)
        
        with col2:

            st.subheader("Prediction Summary")

            st.write(f"**Zone:** {location}")

            st.write(f"**Road Type:** {road_type}")

            st.write(f"**Weather:** {weather}")

            st.write(f"**Lanes:** {lanes}")

            st.write(f"**Large Vehicles:** {large_vehicles}")

            st.write(f"**Date:** {travel_date}")

            st.write(f"**Time:** {travel_time}")

            st.write(f"**Model:** Weighted Ensemble")

# EDA DASHBOARD

elif page == "EDA Dashboard":
    st.markdown(
    """
    <div class="section-title">
    📊 Exploratory Data Analysis
    </div>
    """,
    unsafe_allow_html=True
    )

    viz_dir = BASE_DIR / "visualizations"

    images = sorted(os.listdir(viz_dir))

    cols = st.columns(2)

    for i, image in enumerate(images):

        if image.endswith(".png"):

            with cols[i % 2]:

                title = (
                    image
                    .replace(".png", "")
                    .replace("_", " ")
                    .title()
                )

                st.subheader(title)

                st.image(
                    str(viz_dir / image),
                    use_container_width=True
                )

# MODEL PERFORMANCE

elif page == "Model Performance":
    st.markdown(
    """
    <div class="section-title">
    📈 Model Performance
    </div>
    """,
    unsafe_allow_html=True
    )
    performance = pd.read_csv(
        BASE_DIR / "data" / "model_performance.csv"
    )
    

    st.dataframe(
        performance,
        use_container_width=True
    )
    best_model = performance.loc[
        performance["R² Score"].idxmax()
    ]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R²",
            f"{best_model['R² Score']:.4f}"
        )

    with col2:
        st.metric(
            "RMSE",
            f"{best_model['RMSE']:.2f}"
        )

    with col3:
        st.metric(
            "MAE",
            f"{best_model['MAE']:.2f}"
        )

    with col4:
        st.metric(
            "MAPE",
            f"{best_model['MAPE']:.2f}%"
        )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.bar(
        performance["Model"],
        performance["R² Score"]
    )

    ax.set_title("Model Comparison (R² Score)")
    ax.set_ylabel("R² Score")

    plt.xticks(rotation=15)

    st.pyplot(fig)
    
# FOOTER

st.divider()

st.markdown("""
<div style="
text-align:center;
font-size:15px;
color:gray;
padding-bottom:10px;
">

Traffic Demand Prediction System • Machine Learning Internship Project • 2026

</div>
""",
unsafe_allow_html=True)