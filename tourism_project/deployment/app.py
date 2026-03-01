import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Download and load the model
@st.cache_resource
def load_model():
    # Ensure HF_TOKEN is set for hf_hub_download if repo is private or rate-limited
    # In a deployed Streamlit app, this might come from environment variables or Streamlit secrets
    # os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE" # Uncomment and set if needed
    model_path = hf_hub_download(repo_id="wahedali025/tourism_project", filename="best_tourism_project_model_v1.joblib")
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("Wellness Tourism Package Purchase Predictor")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Input fields for customer details
# Customer Details
age = st.slider('Age', 18, 90, 30)
type_of_contact_options = ['Company Invited', 'Self Inquiry']
type_of_contact_selected = st.selectbox('Type of Contact', type_of_contact_options)
type_of_contact = 0 if type_of_contact_selected == 'Company Invited' else 1 # Assuming LabelEncoder encoding
city_tier = st.selectbox('City Tier', [1, 2, 3])
occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Freelancer', 'Large Business', 'Unemployed'])
gender = st.selectbox('Gender', ['Male', 'Female'])
number_of_person_visiting = st.slider('Number of Persons Visiting', 1, 6, 2)
preferred_property_star = st.slider('Preferred Property Star', 1, 5, 3)
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
number_of_trips = st.slider('Number of Trips Annually', 0, 20, 3)
passport = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox('Owns Car?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.slider('Number of Children Visiting (below 5)', 0, 5, 0)
monthly_income = st.number_input('Monthly Income', min_value=0.0, value=25000.0)

# Customer Interaction Data
pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', 1, 5, 3)
product_pitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Platinum'])
number_of_followups = st.slider('Number of Follow-ups', 0, 10, 3)
duration_of_pitch = st.slider('Duration of Pitch (minutes)', 0.0, 60.0, 15.0)

# Create a DataFrame from inputs
# The order and names of columns must match the training data after preprocessing
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'NumberOfPersonVisiting': [number_of_person_visiting],
    'NumberOfFollowups': [number_of_followups],
    'PreferredPropertyStar': [preferred_property_star],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [number_of_children_visiting],
    'MonthlyIncome': [monthly_income],
    'Occupation': [occupation],
    'Gender': [gender],
    'ProductPitched': [product_pitched],
    'MaritalStatus': [marital_status]
})

# Perform one-hot encoding on categorical features, similar to training
# Ensure all possible categories are known from training data
categorical_cols = ['Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']

# Create dummy columns for all categories present in training data
# This assumes the model was trained on data with these categories
# and that missing categories are handled by the model's preprocessing pipeline or result in 0s
for col in categorical_cols:
    input_data = pd.concat([input_data.drop(columns=[col]), pd.get_dummies(input_data[col], prefix=col)], axis=1)

# Align columns with training data by adding missing columns as 0
# This is crucial for models that expect a fixed number of features
# We need the columns from Xtrain_encoded that the model was trained on.
# For simplicity, we assume the model was trained on Xtrain_encoded and know its columns.
# In a real scenario, you'd save these column names during training.

# Placeholder for the actual columns from training (replace with actual columns from Xtrain_encoded)
# This is a critical step that needs the exact columns from the training data
# For demonstration, let's manually define some expected columns based on previous Xtrain_encoded output:
expected_columns = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'Occupation_Freelancer', 'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business', 'Occupation_Unemployed',
    'Gender_Female', 'Gender_Male',
    'ProductPitched_Basic', 'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Platinum', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
    'MaritalStatus_Divorced', 'MaritalStatus_Married', 'MaritalStatus_Single', 'MaritalStatus_Unmarried'
]

# Add missing columns to input_data and fill with 0
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure the order of columns matches the training data
input_data = input_data[expected_columns] # Reorder to match the model's expected input feature order

if st.button('Predict Purchase'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader("Prediction Results:")
    if prediction == 1:
        st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"The customer is unlikely to purchase the package. (Probability: {prediction_proba:.2f})")
