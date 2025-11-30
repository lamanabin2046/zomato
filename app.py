# =============================================================
# STREAMLIT DELIVERY PREDICTOR — PREMIUM PRO UI (NO HTML)
# ZOMATO THEME + METRICS + GLASSMORPHISM
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Smart Delivery ETA & Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# ----------------------------------------
# CUSTOM CSS — Glassmorphism Zomato Theme
# ----------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Remove default Streamlit top padding */
.block-container {
    padding-top: 0.8rem !important;
}

/* Background */
[data-testid="stAppViewContainer"] > .main {
    background-image: url('https://images.unsplash.com/photo-1534939561126-855b8675edd7?q=80&w=2069');
    background-size: cover;
    background-position: center;
    backdrop-filter: blur(8px);
}

/* Overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    background-color: rgba(0,0,0,0.70);
    position: absolute;
    width: 100%;
    height: 100%;
    top:0; left:0;
    z-index:-1;
}

/* Header Title */
.big-title {
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(90deg, #E23744, #ff7a7a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-top: 10px !important;
    text-shadow: 0px 0px 10px rgba(255,80,80,0.4);
}

/* Subtitle */
.subtitle {
    font-size: 19px;
    text-align: center;
    color: #ffecec;
    margin-top: -5px !important;
    margin-bottom: 35px;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    box-shadow: 0 0px 25px rgba(0,0,0,0.25);
}

label {
    color: white !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #E23744, #ff5e5e);
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: 700;
    width: 100%;
    transition: 0.25s;
}
.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff5252, #ff7777);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# LOAD MODELS AND DATA
# ----------------------------------------
@st.cache_resource
def load_final():
    df = pd.read_csv("final_data.csv")
    df.columns = [c.lower() for c in df.columns]
    return df

@st.cache_resource
def load_regression_model():
    return joblib.load("final_delivery_time_xgb.pkl")

@st.cache_resource
def load_classification_model():
    return joblib.load("delivery_delay_classifier.pkl")

final_df = load_final()
reg_model = load_regression_model()
clf_model = load_classification_model()

# ----------------------------------------
# HEADER SECTION
# ----------------------------------------
st.markdown('<h1 class="big-title">🚚 Smart Delivery ETA & Delay Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Pro-Level Intelligent Delivery Time Estimation System</p>', unsafe_allow_html=True)

# ----------------------------------------
# INPUT FORM — Glass Card
# ----------------------------------------
with st.container():


    st.subheader("📦 Enter Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        delivery_person_age = st.number_input("Delivery Person Age", 18, 60, 30)
        delivery_person_ratings = st.slider("Delivery Rating", 0.0, 5.0, 4.5)
        vehicle_condition = st.slider("Vehicle Condition (0–3)", 0, 3, 1)
        weather_conditions = st.selectbox("Weather", sorted(final_df["weather_conditions"].dropna().unique()))
        festival = st.selectbox("Festival?", sorted(final_df["festival"].dropna().unique()))

    with col2:
        distance_km = st.number_input("Distance (km)", 0.0, value=5.0)
        multiple_deliveries = st.selectbox("Multiple Deliveries?", [0, 1])
        order_dayofweek = st.selectbox("Order Day (0=Mon)", list(range(7)))
        road_traffic_density = st.selectbox("Traffic", sorted(final_df["road_traffic_density"].dropna().unique()))
        order_month = st.selectbox("Order Month", sorted(final_df["order_month"].dropna().unique()))

    with col3:
        type_of_order = st.selectbox("Order Type", sorted(final_df["type_of_order"].dropna().unique()))
        type_of_vehicle = st.selectbox("Vehicle Type", sorted(final_df["type_of_vehicle"].dropna().unique()))
        city = st.selectbox("City", sorted(final_df["city"].dropna().unique()))
        restaurant_zone = st.selectbox("Restaurant Zone", ["0","1","2","3","4"])
        customer_zone = st.selectbox("Customer Zone", ["0","1","2","3","4"])

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------
# FEATURE ENGINEERING (MATCH TRAINING SCRIPT)
# ----------------------------------------
now = datetime.datetime.now()
current_hour = now.hour
week_of_year = now.isocalendar().week
day_of_month = now.day

is_weekend = 1 if order_dayofweek >= 5 else 0
peak_hours = 1 if ((11 <= current_hour < 14) or (17 <= current_hour < 21)) else 0
rush_hour = 1 if (11 <= current_hour < 15) or (18 <= current_hour < 22) else 0

traffic_map = {"low": 1, "medium": 2, "high": 3, "jam": 4}
traffic_ordinal = traffic_map[road_traffic_density.lower()]

delay_weekend = peak_hours * is_weekend
distance_traffic = distance_km * peak_hours
rating_vehicle = delivery_person_ratings * vehicle_condition
driver_score = delivery_person_ratings * (vehicle_condition / 10)
dist_order_hour = distance_km * current_hour
peak_traffic = peak_hours
multi_peak = multiple_deliveries * peak_hours

# distance bins
if distance_km < 2: distance_bin = 0
elif distance_km < 5: distance_bin = 1
elif distance_km < 10: distance_bin = 2
elif distance_km < 20: distance_bin = 3
else: distance_bin = 4

# part of day
def get_part_of_day(h):
    if 5 <= h < 11: return "morning"
    if 11 <= h < 15: return "lunch"
    if 15 <= h < 18: return "afternoon"
    if 18 <= h < 22: return "evening"
    return "night"

part_of_day = get_part_of_day(current_hour)

# age bins
def detect_age_group(age, bins):
    for b in bins:
        try:
            low, high = map(int, b.split("-"))
            if low <= age <= high:
                return b
        except:
            pass
    return "Unknown"

age_bins = detect_age_group(delivery_person_age, final_df["age_bins"].dropna().unique())

# ----------------------------------------
# BUILD INPUT DATAFRAME
# ----------------------------------------
inp = pd.DataFrame([{
    "distance_km": distance_km,
    "distance_traffic": distance_traffic,
    "driver_score": driver_score,
    "delivery_person_age": delivery_person_age,
    "delivery_person_ratings": delivery_person_ratings,
    "vehicle_condition": vehicle_condition,
    "multiple_deliveries": multiple_deliveries,
    "order_dayofweek": order_dayofweek,
    "week_of_year": week_of_year,
    "day_of_month": day_of_month,
    "hour_of_order": current_hour,
    "rush_hour": rush_hour,
    "traffic_ordinal": traffic_ordinal,
    "rating_vehicle": rating_vehicle,
    "delay_weekend": delay_weekend,
    "dist_order_hour": dist_order_hour,
    "peak_traffic": peak_traffic,
    "multi_peak": multi_peak,
    "peak_hours": peak_hours,
    "is_weekend": is_weekend,
    "weather_conditions": weather_conditions,
    "road_traffic_density": road_traffic_density,
    "type_of_order": type_of_order,
    "type_of_vehicle": type_of_vehicle,
    "festival": festival,
    "city": city,
    "order_month": order_month,
    "age_bins": age_bins,
    "part_of_day": part_of_day,
    "restaurant_zone": restaurant_zone,
    "customer_zone": customer_zone,
    "distance_bin": distance_bin,
}])

# ----------------------------------------
# PREDICT & DISPLAY RESULTS — NO HTML
# ----------------------------------------
st.write("")
st.write("")

if st.button("🚀 Predict Delivery Status"):
    try:
        eta = reg_model.predict(inp)[0]
        delay_proba = clf_model.predict_proba(inp)[0][1]

        delay_label = "Delayed" if delay_proba > 0.5 else "On-Time"
        delay_icon = "❗" if delay_proba > 0.5 else "✔"

        # VERY CLEAN OUTPUT AREA
        st.subheader("📊 Prediction Results")

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### ⏱ Estimated Delivery Time")
            st.metric(label="ETA (minutes)", value=f"{eta:.2f}")

        with colB:
            st.markdown("### 🚦 Delivery Status")
            st.metric(label="Status", value=f"{delay_icon} {delay_label}")
            st.caption(f"Probability of Delay: {delay_proba:.2f}")

    except Exception as e:
        st.error("Error:")
        st.code(str(e))
