import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from lightgbm import LGBMRegressor
# Load the California housing dataset
dataset = fetch_california_housing()
df = pd.DataFrame(data=dataset['data'], columns=dataset['feature_names'])

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# App UI
st.title("California House Price Prediction App 🏠")
st.write("Enter the features below to predict the **Median House Value** (in $100,000s).")

# Feature input widgets
col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income (10k USD)", 0.0, 20.0, 3.0)
    HouseAge = st.slider("House Age (years)", 1, 60, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("Population", 100, 5000, 1000)
    AveOccup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# Prepare input
input_data = pd.DataFrame({
    "MedInc": [MedInc],
    "HouseAge": [HouseAge],
    "AveRooms": [AveRooms],
    "AveBedrms": [AveBedrms],
    "Population": [Population],
    "AveOccup": [AveOccup],
    "Latitude": [Latitude],
    "Longitude": [Longitude]
})

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Predicted Median House Value: **${prediction * 100000:.2f}**")

