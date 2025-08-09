import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("crop_yield_model.pkl")
le_crop = joblib.load("le_crop.pkl")
le_district = joblib.load("le_district.pkl")
le_yield_category = joblib.load("le_yield_category.pkl")
df = pd.read_csv("historical_crop_yield_dataset.csv")

# App UI
st.set_page_config(page_title="Smart Crop Yield Predictor", layout="centered")
st.title("\U0001F33E Smart Crop Yield Predictor")

st.sidebar.title("👨‍🌾 Farmer Input")
district = st.sidebar.selectbox("Select District", sorted(df["District"].unique()))
ph = st.sidebar.slider("Enter Soil pH Level", 3.5, 9.0, 6.5)
year = st.sidebar.selectbox("Prediction Year", [2025, 2026])
previous_year = st.sidebar.selectbox("Previous Year (for analysis)", sorted(df["Year"].unique(), reverse=True))

# Get auto data for district and year
def get_avg_values(district, year):
    past = df[(df["District"] == district) & (df["Year"] >= year - 10) & (df["Year"] < year)]
    return {
        "rainfall": round(past["Rainfall (mm)"].mean(), 2),
        "temperature": round(past["Temperature (°C)"].mean(), 2),
        "fertilizer": round(past["Fertilizer Used (kg/ha)"].mean(), 2)
    }

auto_data = get_avg_values(district, year)

# Buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("📈 Predict Future Crop Yield")
with col2:
    analyze_btn = st.button("📊 Analyze Previous Year Yield")

# Prediction Logic
if predict_btn:
    st.subheader("🌟 Crop Yield Predictions for {}".format(year))
    results = []
    for crop in df["Crop"].unique():
        input_data = pd.DataFrame([{
            "DistrictEncoded": le_district.transform([district])[0],
            "Year": year,
            "Rainfall (mm)": auto_data["rainfall"],
            "Temperature (°C)": auto_data["temperature"],
            "Fertilizer Used (kg/ha)": auto_data["fertilizer"],
            "pH Level": ph
        }])
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = le_yield_category.inverse_transform([prediction_encoded])[0]
        results.append((crop, prediction_label))

    # Show results in chart
    good, avg, bad = 0, 0, 0
    for crop, pred in results:
        emoji = "✅" if pred == "Good" else "⚠️" if pred == "Average" else "❌"
        st.write(f"{emoji} **{crop}** → {pred}")
        if pred == "Good": good += 1
        elif pred == "Average": avg += 1
        else: bad += 1

    # Pie chart
    labels = ['Good', 'Average', 'Bad']
    values = [good, avg, bad]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

if analyze_btn:
    st.subheader(f"📉 Yield Analysis for {previous_year}")
    data = df[(df["District"] == district) & (df["Year"] == previous_year)]
    if data.empty:
        st.warning("No data available for the selected year and district.")
    else:
        st.dataframe(data[["Crop", "Yield Category", "Rainfall (mm)", "Temperature (°C)", "Fertilizer Used (kg/ha)"]])
        st.success(f"{len(data)} records found for {district} in {previous_year}")

# Footer/About
st.markdown("---")
st.markdown("✅ **Project by ABINAYA R, BALADARANESH, SAKTHIKRISHNA G.**")