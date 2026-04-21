import streamlit as st
import pandas as pd
import pickle

# load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Prediksi Biaya Medis 🏥")

# =========================
# INPUT USER
# =========================

age = st.number_input("Umur", 0, 100, 30)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
daily_steps = st.number_input("Langkah Harian", 0, 20000, 5000)
sleep_hours = st.number_input("Jam Tidur", 0.0, 24.0, 7.0)

doctor_visits_per_year = st.number_input("Kunjungan Dokter per Tahun", 0, 20, 2)
hospital_admissions = st.number_input("Rawat Inap per Tahun", 0, 10, 1)
medication_count = st.number_input("Jumlah Obat", 0, 20, 2)
insurance_coverage_pct = st.number_input("Cakupan Asuransi (%)", 0, 100, 75)

# kategori
gender = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
asthma = st.selectbox("Asthma", ["Yes", "No"])

physical_activity_level = st.selectbox("Physical Activity", ["Low", "Medium", "High"])
stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
city_type = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])

# =========================
# PREDIKSI
# =========================
if st.button("Prediksi"):

    # 1. Data lengkap
    data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "smoker": [smoker],
        "diabetes": [diabetes],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "asthma": [asthma],
        "physical_activity_level": [physical_activity_level],
        "daily_steps": [daily_steps],
        "sleep_hours": [sleep_hours],
        "stress_level": [stress_level],
        "doctor_visits_per_year": [doctor_visits_per_year],
        "hospital_admissions": [hospital_admissions],
        "medication_count": [medication_count],
        "insurance_coverage_pct": [insurance_coverage_pct],
        "city_type": [city_type]
    })

    # 2. encoding (SAMA seperti training)
    data = pd.get_dummies(data, drop_first=True)

    # 3. samakan kolom
    data = data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # 4. scaling
    data_scaled = scaler.transform(data)

    # 5. prediksi
    hasil = model.predict(data_scaled)

    st.success(f"Hasil Prediksi Biaya Medis: ${hasil[0]:,.2f}")