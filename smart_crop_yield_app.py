# smart_crop_yield_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64
import numpy as np
import io

# -----------------------------------------------------------------------------
# Load the trained model and all necessary encoders
# -----------------------------------------------------------------------------
try:
    model = joblib.load("crop_yield_model.pkl")
    le_crop = joblib.load("le_crop.pkl")
    le_district = joblib.load("le_district.pkl")
    le_yield_category = joblib.load("le_yield_category.pkl")
    df = pd.read_csv("historical_crop_yield_dataset.csv")
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure all necessary files (.pkl and .csv) are in the same directory.")
    st.stop()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

# Function to get average historical values for a given district for prediction
def get_avg_values(district, year):
    """Calculates the 10-year average values for a district, or defaults to 2014-2024 if no data exists."""
    past = df[(df["District"] == district) & (df["Year"] >= year - 10) & (df["Year"] < year)]
    if past.empty:
        # Fallback to a different date range if not enough historical data is available
        past = df[(df["Year"] >= 2014) & (df["Year"] < 2024)]
    
    return {
        "rainfall": round(past["Rainfall (mm)"].mean(), 2) if not past.empty else 0,
        "temperature": round(past["Temperature (°C)"].mean(), 2) if not past.empty else 0,
        "fertilizer": round(past["Fertilizer Used (kg/ha)"].mean(), 2) if not past.empty else 0
    }

# Function to get specific year's average values for a district for analysis
def get_specific_year_values(district, year):
    """Calculates the average values for a specific year in a district."""
    data = df[(df["District"] == district) & (df["Year"] == year)]
    if data.empty:
        return {
            "rainfall": 0,
            "temperature": 0,
            "fertilizer": 0
        }
    else:
        return {
            "rainfall": round(data["Rainfall (mm)"].mean(), 2),
            "temperature": round(data["Temperature (°C)"].mean(), 2),
            "fertilizer": round(data["Fertilizer Used (kg/ha)"].mean(), 2)
        }

# Function to get a predicted numerical yield based on the category
def get_predicted_yield(prediction_label, crop):
    """
    Estimates a numerical yield in kg/acre based on the categorical prediction.
    This function uses historical data to find an average yield for the given
    category and crop, falling back to a global average if necessary.
    """
    # 1 hectare = 2.47105 acres, 1 ton = 1000 kg
    ton_per_ha_to_kg_per_acre = 1000 / 2.47105

    # Filter data for the specific crop and prediction label
    filtered_data = df[(df["Crop"] == crop) & (df["Yield Category"] == prediction_label)]
    
    # Calculate the average yield for the specific crop and category, if available
    if not filtered_data.empty:
        avg_yield_ton_per_ha = filtered_data["Previous Year Yield (ton/ha)"].mean()
    else:
        # Fallback to a global average for the category if the specific crop isn't found
        global_avg_yield = df[df["Yield Category"] == prediction_label]["Previous Year Yield (ton/ha)"].mean()
        avg_yield_ton_per_ha = global_avg_yield if not pd.isna(global_avg_yield) else 0

    # Convert to kg/acre
    avg_yield_kg_per_acre = avg_yield_ton_per_ha * ton_per_ha_to_kg_per_acre
    return round(avg_yield_kg_per_acre, 2)


# Function to generate a human-readable reason for the prediction
def generate_reason(prediction, rainfall, temperature, ph):
    if prediction == "Good":
        return "The combination of balanced pH levels and optimal rainfall and temperature conditions is ideal for a high yield."
    elif prediction == "Average":
        return "The conditions are adequate, but factors like rainfall or temperature are slightly below the ideal range, leading to an average yield."
    else: # Bad
        reasons = []
        if rainfall < 1000:
            reasons.append("low rainfall")
        if temperature > 35:
            reasons.append("high temperature")
        if ph < 6.0 or ph > 7.5:
            reasons.append("unbalanced pH level")
        
        reason_text = " and ".join(reasons)
        if reason_text:
            return f"The yield is low due to {reason_text}."
        else:
            return "The conditions are not ideal for this crop."

# Function to create a downloadable report
def get_download_link(text_content):
    b64 = base64.b64encode(text_content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="crop_yield_report.txt">📄 Download Predicted Report</a>'
    return href

# -----------------------------------------------------------------------------
# Streamlit App User Interface
# -----------------------------------------------------------------------------

# Set up the page configuration for a professional look
st.set_page_config(page_title="Smart Crop Yield Predictor", layout="centered")

# Custom CSS for a professional and attractive look
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stMarkdown h1 {
        color: #006400;
        text-align: center;
        font-size: 2.5rem;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar .stSelectbox {
        color: #006400;
        font-weight: bold;
    }
    .navbar-container {
        background: #f1f8e9; /* A light, decent green background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 3px solid #81c784;
    }
    .navbar-container h3 {
        color: #333; /* A dark grey color for better contrast */
        font-size: 1.2rem;
        font-weight: bold;
    }
    .team-names {
        margin-top: 10px;
        color: #555;
        font-size: 1rem;
        font-style: italic;
    }
    .auto-data-container {
        border-left: 5px solid #2e7d32; /* Darker green border */
        background-color: #e8f5e9; /* Very light green background */
        color: #000000; /* Black text for visibility */
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-family: 'Georgia', serif; /* A formal serif font */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle shadow for depth */
    }
    .auto-data-container b {
        font-family: 'Arial', sans-serif; /* Keep the bold text in a clean font */
        font-size: 1.1em;
    }
    .auto-data-container ul {
        list-style-type: none; /* Remove bullet points */
        padding: 0;
    }
    .auto-data-container li {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main App Header as a "navbar"
st.markdown("""
<div class="navbar-container">
    <h1>🌾 Smart Crop Yield Predictor</h1>
    <h3>Your one-stop solution for smart agricultural decisions.</h3>
    <p class="team-names">Project by ABINAYA R, BALADARANESH, SAKTHIKRISHNA G.</p>
</div>
""", unsafe_allow_html=True)


# Sidebar for user input
st.sidebar.title("👨‍🌾 Farmer Input")
district = st.sidebar.selectbox("Select District", sorted(df["District"].unique()))
crop = st.sidebar.selectbox("Select Crop", sorted(df["Crop"].unique()))
ph = st.sidebar.slider("Enter Soil pH Level", 3.5, 9.0, 6.5)

# Prediction year and previous year for analysis, limited to 2025 and 2026
prediction_year = st.sidebar.selectbox("Prediction Year", [2025, 2026])
previous_year = st.sidebar.selectbox("Previous Year (for analysis)", sorted(df["Year"].unique(), reverse=True))

# Main app logic and buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("📈 Predict Future Crop Yield")
with col2:
    analyze_btn = st.button("📊 Analyze Previous Year Yield")

# Initialize state variables
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'show_download' not in st.session_state:
    st.session_state.show_download = False
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'result_type' not in st.session_state:
    st.session_state.result_type = None

# Logic for displaying results
if predict_btn:
    st.session_state.show_results = True
    st.session_state.result_type = "prediction"
    st.session_state.show_download = True

elif analyze_btn:
    st.session_state.show_results = True
    st.session_state.result_type = "analysis"
    st.session_state.show_download = False

# Display results if a button has been pressed
if st.session_state.show_results:
    if st.session_state.result_type == "prediction":
        auto_data = get_avg_values(district, prediction_year)
        data_title = "Auto-Filled Data (10-Year Average)"
        
        # Display Auto-Filled Data
        st.markdown("---")
        st.markdown(f"""
        <div class="auto-data-container">
            <b>{data_title}:</b>
            <ul>
                <li><b>Rainfall:</b> {auto_data["rainfall"]} mm</li>
                <li><b>Temperature:</b> {auto_data["temperature"]} °C</li>
                <li><b>Fertilizer:</b> {auto_data["fertilizer"]} kg/ha</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(f"🌟 Crop Yield Prediction for {crop} in {district} ({prediction_year})")

        report_text = f"Crop Yield Prediction Report for {crop} in {district} - {prediction_year}\n\n"
        report_text += f"Input Parameters:\n"
        report_text += f"------------------\n"
        report_text += f"pH Level: {ph}\n"
        report_text += f"Average Rainfall: {auto_data['rainfall']} mm\n"
        report_text += f"Average Temperature: {auto_data['temperature']} °C\n"
        report_text += f"Average Fertilizer: {auto_data['fertilizer']} kg/ha\n\n"
        report_text += "Prediction Results:\n"
        report_text += "-------------------\n\n"

        # Prepare input data for the model with the selected crop
        input_data = pd.DataFrame([{
            "DistrictEncoded": le_district.transform([district])[0],
            "CropEncoded": le_crop.transform([crop])[0],
            "pH Level": ph,
            "Year": prediction_year,
            "Rainfall (mm)": auto_data["rainfall"],
            "Temperature (°C)": auto_data["temperature"],
            "Fertilizer Used (kg/ha)": auto_data["fertilizer"]
        }])
        
        # Predict and get the human-readable label
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = le_yield_category.inverse_transform([prediction_encoded])[0]
        
        # Get the numerical yield in kg/acre
        predicted_yield_kg_per_acre = get_predicted_yield(prediction_label, crop)
        
        # Generate the explanation
        reason = generate_reason(prediction_label, auto_data["rainfall"], auto_data["temperature"], ph)
        
        # Show results and build report text
        emoji = "✅" if prediction_label == "Good" else "⚠️" if prediction_label == "Average" else "❌"
        st.write(f"{emoji} **{crop}** → {prediction_label} (Estimated: **{predicted_yield_kg_per_acre} kg/acre**)")
        st.markdown(f"> *Reason:* {reason}")
        
        report_text += f"{emoji} {crop} -> {prediction_label}\n"
        report_text += f"Estimated Yield: {predicted_yield_kg_per_acre} kg/acre\n"
        report_text += f"Reason: {reason}\n\n"
        
        st.session_state.report_content = report_text
    
    elif st.session_state.result_type == "analysis":
        # Filter data for the specific crop, year, and district
        data = df[(df["District"] == district) & (df["Year"] == previous_year) & (df["Crop"] == crop)]
        
        # Get specific year data for the district (I'll keep this as it's useful for context)
        specific_year_data = get_specific_year_values(district, previous_year)
        
        data_title = f"Data for {previous_year}"
        
        # Display Auto-Filled Data
        st.markdown("---")
        st.markdown(f"""
        <div class="auto-data-container">
            <b>{data_title}:</b>
            <ul>
                <li><b>Rainfall:</b> {specific_year_data["rainfall"]} mm</li>
                <li><b>Temperature:</b> {specific_year_data["temperature"]} °C</li>
                <li><b>Fertilizer:</b> {specific_year_data["fertilizer"]} kg/ha</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader(f"📉 Yield Analysis for {crop} in {district} ({previous_year})")
        
        if data.empty:
            st.warning(f"No data available for {crop} in {district} for the year {previous_year}.")
        else:
            # Get the single record (assuming there's only one per crop/district/year)
            record = data.iloc[0]
            
            # Generate a narrative summary of the specific record
            narrative = (
                f"In the year {int(record['Year'])}, the yield for {record['Crop']} in {record['District']} was categorized as **{record['Yield Category']}**. "
                f"The key factors recorded were: "
                f"a rainfall of **{record['Rainfall (mm)']} mm**, "
                f"a temperature of **{record['Temperature (°C)']}°C**, "
                f"and a soil pH level of **{record['pH Level']}**. "
                f"The previous year's yield was **{record['Previous Year Yield (ton/ha)']} tons/ha**, "
                f"and **{record['Fertilizer Used (kg/ha)']} kg/ha** of fertilizer was used."
            )
            st.info(narrative)

    # Show the download button only after a prediction is made
    if st.session_state.show_download:
        st.markdown("---")
        st.markdown(get_download_link(st.session_state.report_content), unsafe_allow_html=True)

# Footer/About
st.markdown("---")
