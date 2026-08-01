"""
app.py – Wellness Tourism Package Predictor (Streamlit Application)
===================================================================
A premium Streamlit web application that predicts whether a customer
will purchase the Wellness Tourism Package. Loads the trained sklearn
Pipeline (preprocessor + XGBoost) from the same directory and provides
an interactive interface for real-time predictions.

Deployed on Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# ─── 1. Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Wellness Tourism Predictor | Visit With Us",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 2. Custom CSS for Premium UI ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Header with Shimmer Animation ── */
    .hero-header {
        background: linear-gradient(135deg, #0d9488 0%, #065f46 50%, #064e3b 100%);
        background-size: 200% 200%;
        animation: shimmer 6s ease infinite;
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(13, 148, 136, 0.3);
    }
    @keyframes shimmer {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-title {
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-weight: 400;
        font-size: 1.1rem;
        opacity: 0.85;
    }

    /* ── Info Cards ── */
    .info-card {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border-left: 5px solid #0d9488;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
    }
    .info-card-label {
        font-size: 0.78rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.15rem;
    }
    .info-card-value {
        font-size: 1.15rem;
        color: #111827;
        font-weight: 700;
    }

    /* ── Prediction Cards ── */
    .pred-card {
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .pred-card:hover {
        transform: translateY(-5px);
    }
    .pred-yes {
        background: linear-gradient(135deg, #059669, #10b981);
        border: 2px solid #34d399;
    }
    .pred-no {
        background: linear-gradient(135deg, #b91c1c, #ef4444);
        border: 2px solid #fca5a5;
    }
    .pred-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .pred-prob {
        color: white;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .pred-label {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        font-weight: 400;
    }

    /* ── Confidence Progress Bar ── */
    .conf-container {
        width: 100%;
        background: rgba(255,255,255,0.25);
        border-radius: 999px;
        height: 10px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        background: rgba(255,255,255,0.85);
        border-radius: 999px;
        transition: width 1s ease-in-out;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdfa, #ecfdf5);
        border-right: 1px solid #d1fae5;
    }
    .sidebar-section {
        font-size: 1.1rem;
        font-weight: 700;
        color: #065f46;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #a7f3d0;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── 3. Load Trained Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained sklearn Pipeline from the same directory."""
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ─── 4. Hero Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🌿 Wellness Tourism Package Predictor</div>
    <div class="hero-subtitle">Advanced ML-driven insights for customer conversion | Visit With Us™</div>
</div>
""", unsafe_allow_html=True)


# ─── 5. Sidebar — Organized Customer Inputs ──────────────────────────────
with st.sidebar:
    st.markdown("## 🧑‍💼 Customer Profile")

    # ── Demographics Section ──
    st.markdown('<div class="sidebar-section">📋 Demographics</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=100000, value=22000, step=500)

    # ── Travel Profile Section ──
    st.markdown('<div class="sidebar-section">✈️ Travel Profile</div>', unsafe_allow_html=True)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes 🚗" if x == 1 else "No")
    num_trips = st.number_input("Avg. Annual Trips", min_value=1, max_value=25, value=3, step=1)
    num_persons = st.number_input("Persons Visiting", min_value=1, max_value=5, value=3, step=1)
    num_children = st.number_input("Children (< 5 yrs)", min_value=0, max_value=3, value=1, step=1)
    preferred_star = st.selectbox("Preferred Hotel Star", [3, 4, 5])

    # ── Interaction Data Section ──
    st.markdown('<div class="sidebar-section">💬 Interaction Data</div>', unsafe_allow_html=True)
    type_of_contact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_pitch = st.number_input("Pitch Duration (min)", min_value=5, max_value=130, value=15, step=1)
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=4, step=1)
    pitch_score = st.selectbox("Pitch Satisfaction Score (1-5)", [1, 2, 3, 4, 5], index=2)


# ─── 6. Build Input DataFrame ────────────────────────────────────────────
input_data = {
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}
input_df = pd.DataFrame([input_data])


# ─── Helper: Create Info Card HTML ────────────────────────────────────────
def info_card(label, value):
    return f"""
    <div class="info-card">
        <div class="info-card-label">{label}</div>
        <div class="info-card-value">{value}</div>
    </div>"""


# ─── 7. Main Content — Two Column Layout ─────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Customer Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(info_card("Age", f"{age} yrs"), unsafe_allow_html=True)
        st.markdown(info_card("Occupation", occupation), unsafe_allow_html=True)
    with c2:
        st.markdown(info_card("Income", f"${monthly_income:,}"), unsafe_allow_html=True)
        st.markdown(info_card("Designation", designation), unsafe_allow_html=True)
    with c3:
        st.markdown(info_card("Passport", "Yes ✅" if passport else "No ❌"), unsafe_allow_html=True)
        st.markdown(info_card("Product Pitched", product_pitched), unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Prediction Result")

    if not model_loaded:
        st.error(f"⚠️ Model not loaded. Please ensure the CI/CD pipeline has run successfully.\n\nError: {model_error}")
    else:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:
            conf = probability[1] * 100
            st.markdown(f"""
            <div class="pred-card pred-yes">
                <div class="pred-title">✅ Likely to Purchase</div>
                <div class="pred-prob">{conf:.1f}%</div>
                <div class="pred-label">Purchase Confidence</div>
                <div class="conf-container">
                    <div class="conf-fill" style="width:{conf}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.success("💡 **Recommendation:** This customer is a high-potential lead. Prioritize outreach and consider premium add-on offers!")
        else:
            conf = probability[0] * 100
            st.markdown(f"""
            <div class="pred-card pred-no">
                <div class="pred-title">❌ Unlikely to Purchase</div>
                <div class="pred-prob">{conf:.1f}%</div>
                <div class="pred-label">Non-Purchase Confidence</div>
                <div class="conf-container">
                    <div class="conf-fill" style="width:{conf}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 **Recommendation:** Nurture this lead with standard marketing. Consider alternative package offerings or adjust the pitch strategy.")


# ─── 8. Expandable Input Data View ───────────────────────────────────────
with st.expander("📋 View Complete Input Data", expanded=False):
    st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ─── 9. Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <p>🌿 <strong>Visit With Us</strong> — Wellness Tourism Division</p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        Powered by XGBoost & Streamlit | MLOps Pipeline on GitHub Actions<br>
        © 2026 Visit With Us. Model predictions are guidance — use alongside sales expertise.
    </p>
</div>
""", unsafe_allow_html=True)
