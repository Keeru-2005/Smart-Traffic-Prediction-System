import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Display the title
st.title("Smart Traffic Prediction System")

# Add a subtle note about the dataset and model
st.markdown("""
    *Note: This model is still a work in progress and has been trained on a traffic volume dataset sourced from Kaggle.*
""")

# Streamlit UI for user input
hour = st.slider("Hour of the Day", 0, 23, 9)
clouds_all = st.slider("Cloud Coverage (%)", 0, 100, 88)

# Add note near the "Day of the Week" slider
st.markdown("**Note**: 0 is Monday and 6 is Sunday.")
day_of_week = st.slider("Day of the Week", 0, 6, 2)  # 0: Monday, 6: Sunday

# Prepare input features for prediction (removed temperature)
input_features = pd.DataFrame([[hour, clouds_all, day_of_week]], columns=['hour', 'clouds_all', 'day_of_week'])

# Scale the input features
input_features_scaled = scaler.transform(input_features)

# Make the prediction
predicted_log_traffic = model.predict(input_features_scaled)[0]
predicted_traffic = np.exp(predicted_log_traffic) - 1  # Reverse the log transformation

# Ensure the predicted traffic is non-negative
predicted_traffic = max(0, predicted_traffic)

# Display the predicted traffic volume
st.write(f"Predicted Traffic Volume: {predicted_traffic:.2f}")

# Classify the traffic volume into High, Medium, or Low
if predicted_traffic < 1000:
    traffic_status = "Low Traffic"
elif 1000 <= predicted_traffic < 3000:
    traffic_status = "Medium Traffic"
else:
    traffic_status = "High Traffic"

# Display the traffic status
st.write(f"Traffic Status: {traffic_status}")
