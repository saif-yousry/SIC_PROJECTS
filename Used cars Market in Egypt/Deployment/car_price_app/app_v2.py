import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("### Predict the price of your car based on its features")

# Load everything
@st.cache_resource
def load_assets():
    with open('car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('unique_values.pkl', 'rb') as f:
        unique_vals = pickle.load(f)
    return model, encoders, scaler, unique_vals

try:
    model, encoders, scaler, unique_vals = load_assets()
except:
    st.error("⚠️ Required files missing! Make sure you have: car_price_model.pkl, label_encoders.pkl, scaler.pkl, unique_values.pkl")
    st.stop()

# UI
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", unique_vals['brands'])
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2020, step=1)
    color = st.selectbox("Color", unique_vals['colors'])

with col2:
    car_model = st.text_input("Model", placeholder="e.g., Camry")
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000)
    transmission = st.selectbox("Transmission", unique_vals['transmissions'])

st.markdown("---")

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    if not car_model:
        st.warning("⚠️ Please enter a car model.")
    else:
        current_year = 2025
        car_age = current_year - year
        mileage_per_year = mileage / (car_age + 1)
        
        luxury = ['Mercedes', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Land Rover', 'Lexus']
        premium = ['Volkswagen', 'Volvo', 'Mazda', 'Subaru', 'Infiniti']
        brand_category = 'Luxury' if brand in luxury else ('Premium' if brand in premium else 'Standard')
        
        # Create input matching training data
        input_df = pd.DataFrame({
            'Brand': [brand],
            'Model': [car_model],
            'Year': [year],
            'Milage': [mileage],
            'Transmission': [transmission],
            'Color': [color],
            'Fuel': ['Petrol'],
            'Car_Age': [car_age],
            'Mileage_Per_Year': [mileage_per_year],
            'Brand_Category': [brand_category]
        })
        
        # label encoding
        for col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # Apply scaling to numerical columns
        num_cols = list(scaler.feature_names_in_)
        input_df.loc[:, num_cols] = scaler.transform(input_df[num_cols])

        model_cols = list(model.feature_names_in_)
        input_df = input_df[model_cols]

        prediction = model.predict(input_df)[0]
        
        min_val = prediction * 0.9
        max_val = prediction * 1.1
        
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"${prediction:,.2f}")
        col2.metric("Min Value", f"${min_val:,.2f}")
        col3.metric("Max Value", f"${max_val:,.2f}")
