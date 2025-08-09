import streamlit as st
import pandas as pd

# Set the title of the app
st.title("Historical Crop Yield Data Analysis")

# Use a text input for the user to enter the file name
file_name = st.text_input("Enter the CSV file name:", "historical_crop_yield_dataset.csv")

# Try to load the data from the CSV file
try:
    df = pd.read_csv(file_name)
    st.success(f"Successfully loaded data from {file_name}!")

    # Display the DataFrame
    st.subheader("Raw Data")
    st.dataframe(df)

    # You can add more features here, like charts or filters.
    st.subheader("Data Overview")
    st.write(f"The dataset contains **{len(df)}** records with **{len(df.columns)}** columns.")

    # Show a simple bar chart of records per crop
    st.subheader("Records per Crop")
    crop_counts = df['Crop'].value_counts()
    st.bar_chart(crop_counts)

except FileNotFoundError:
    st.error(f"Error: The file '{file_name}' was not found. Please make sure it's in the same directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
