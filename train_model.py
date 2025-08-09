# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the historical crop yield dataset from the CSV file
df = pd.read_csv("historical_crop_yield_dataset.csv")

# Drop any rows with missing values to ensure clean data for training
df.dropna(inplace=True)

# -----------------------------------------------------------------------------
# Data Preprocessing and Feature Engineering
# -----------------------------------------------------------------------------

# Initialize LabelEncoders for categorical features and the target variable
le_crop = LabelEncoder()
le_district = LabelEncoder()
le_yield_category = LabelEncoder()

# Encode the categorical features and add them as new columns to the DataFrame
df["CropEncoded"] = le_crop.fit_transform(df["Crop"])
df["DistrictEncoded"] = le_district.fit_transform(df["District"])

# Encode the target variable 'Yield Category'
df["YieldCategoryEncoded"] = le_yield_category.fit_transform(df["Yield Category"])

# -----------------------------------------------------------------------------
# Model Training
# -----------------------------------------------------------------------------

# Define the features (X) and the target variable (y) for the model.
# NOTE: The 'CropEncoded' column is now included to ensure varied predictions.
X = df[[
    "DistrictEncoded",
    "CropEncoded",  # <--- NEW: Now includes the crop as a feature
    "pH Level",
    "Year",
    "Rainfall (mm)",
    "Temperature (°C)",
    "Fertilizer Used (kg/ha)"
]]

y = df["YieldCategoryEncoded"]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Saving the Model and Encoders
# -----------------------------------------------------------------------------

# Save the trained model and all the encoders using joblib
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(le_crop, "le_crop.pkl")
joblib.dump(le_district, "le_district.pkl")
joblib.dump(le_yield_category, "le_yield_category.pkl")

print("✅ Model trained and saved successfully with all encoders.")
print("The model is now trained with the 'Crop' as a feature, which will provide more accurate and specific predictions.")
