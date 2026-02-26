import streamlit as st
from PIL import Image
import joblib
import numpy as np

# ===============================
# Load saved model & preprocessors
# ===============================
st.write("App loaded successfully")
model = joblib.load("best_model.pkl")
scaler = joblib.load("sc.pkl")
label_encoder = joblib.load("le.pkl")   

# ===============================
# Title & Header
# ===============================
st.title("🌱 AgriGuide – Crop Recommendation System")
st.header("ML Based Agricultural Analysis")
st.subheader("Enter Soil & Climate Details")

st.write(
    "This application predicts the **most suitable crop** "
    "based on soil nutrients, climate conditions and season."
)
st.image(Image.open(r"C:\Users\anamika\Desktop\DS\VS Code\ML\ML Deployment\Image-Crop-Recommendation@1x-1.png"))
# ===============================
# User Inputs
# ===============================
N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
ph = st.number_input("Soil pH", step=0.1)

rainfall = st.number_input("Rainfall (mm)", step=0.1)
temperature = st.number_input("Temperature (°C)", step=0.1)
humidity = st.number_input("Humidity (%)", step=0.1)

# ===============================
# Season Mapping (training match)
# ===============================
season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])

season_map = {
    "Kharif": 0,
    "Rabi": 1,
    "Zaid": 2
}

# ===============================
# Prediction
# ===============================
if st.button("Predict Crop 🌾"):
    try:
        season_encoded = season_map[season]

        # EXACT training order
        input_data = np.array([[
            N,
            P,
            K,
            ph,
            rainfall,
            temperature,
            humidity,
            season_encoded
        ]])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediction (number → crop name)
        pred_num = model.predict(input_scaled)[0]
        pred_crop = label_encoder.inverse_transform([pred_num])[0]

        st.success(f"🌾 Recommended Crop: **{pred_crop}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
