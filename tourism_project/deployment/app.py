import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title="Wellness Tourism Predictor", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .hero-header { background: linear-gradient(135deg, #0d9488, #064e3b); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
    .pred-card { padding: 2rem; border-radius: 15px; text-align: center; color: white; }
    .pred-yes { background: #059669; }
    .pred-no { background: #b91c1c; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    return joblib.load(model_path)

st.markdown('<div class="hero-header"><h1>🌿 Wellness Tourism Predictor</h1><p>Visit With Us™ - Internal Sales Tool</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🧑‍💼 Customer Profile")
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occ = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    desig = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income ($)", 1000, 100000, 22000)
    st.divider()
    st.header("✈️ Travel & Interaction")
    tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    car = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    trips = st.number_input("Annual Trips", 1, 20, 3)
    pitch_dur = st.number_input("Pitch Duration (min)", 5, 120, 15)
    followups = st.number_input("Follow-ups", 1, 6, 4)
    pitch_score = st.slider("Satisfaction Score", 1, 5, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    persons = st.number_input("Persons Visiting", 1, 10, 2)
    children = st.number_input("Children", 0, 5, 0)
    stars = st.selectbox("Property Stars", [3, 4, 5])
    contact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])

input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": contact, "CityTier": tier, "DurationOfPitch": pitch_dur,
    "Occupation": occ, "Gender": gender, "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
    "ProductPitched": product, "PreferredPropertyStar": stars, "MaritalStatus": marital,
    "NumberOfTrips": trips, "Passport": passport, "PitchSatisfactionScore": pitch_score,
    "OwnCar": car, "NumberOfChildrenVisiting": children, "Designation": desig, "MonthlyIncome": income
}])

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📋 Input Summary")
    st.write(input_df.T)

with col2:
    st.subheader("🎯 Prediction")
    if st.button("Predict Purchase Likelihood", type="primary"):
        try:
            model = load_model()
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            
            if pred == 1:
                st.markdown(f'<div class="pred-card pred-yes"><h2>Likely to Purchase</h2><h1>{prob[1]*100:.1f}% Confidence</h1></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-card pred-no"><h2>Unlikely to Purchase</h2><h1>{prob[0]*100:.1f}% Confidence</h1></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
