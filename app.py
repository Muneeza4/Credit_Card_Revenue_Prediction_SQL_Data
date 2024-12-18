import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Define the prediction function
def predict_revenue(age, gender, job, income):
    input_data = pd.DataFrame({'customer_age': [age], 'gender': [gender], 'customer_job': [job], 'income': [income]})
    predicted_revenue = model.predict(input_data)[0]
    return predicted_revenue

# Title of the app
st.title('Revenue Prediction App')

# Description of the app
st.write("""
    This app predicts the revenue based on customer data such as age, gender, job, and income.
    Please enter the customer information below to predict the revenue.
""")

# Input fields
age = st.number_input("Enter customer age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Select customer gender", ["Male", "Female"])
job = st.text_input("Enter customer job")
income = st.number_input("Enter customer income", min_value=1000, max_value=1000000, step=1000)

# When the user clicks the "Predict" button
if st.button("Predict Revenue"):
    if not job:
        st.error("Please enter a valid job.")
    else:
        predicted_revenue = predict_revenue(age, gender, job, income)
        st.success(f"Predicted Revenue: {predicted_revenue:,.2f}")

# Optional: Show some info about the model
st.sidebar.header('Model Information')
st.sidebar.write("""
    This model uses a Random Forest Regressor to predict revenue.
    The input features include:
    - Age
    - Gender
    - Job
    - Income
""")

# Run the Streamlit app:
# To run the app, use the following command:
# streamlit run app.py