import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import tensorflow as tf

#Load the trained model
model = tf.keras.models.load_model('model.tf')

# Load encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title("Customer Churn Prediction")

#User Input

geography = st.selectbox(
    "Geography",
    options=label_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    options=label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=92
)

balance = st.number_input(
    "Balance")
credit_score = st.number_input(
    "Credit Score")
estimated_salary = st.number_input(
    "Estimated Salary")
tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10
)
num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4
)
has_cr_card = st.selectbox(
    "Has Credit Card",
    options=[0,1]
)
is_active_member = st.selectbox(
    "Is Active Member",
    options=[0,1]
)

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],    
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# Encode geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_scaled_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(["Geography"]))

# Concatenate the geographical features with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_scaled_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write("Prediction Probability:", prediction_proba)

# Convert prediction to binary outcome
if prediction_proba > 0.5:
    prediction_sent = "The customer is likely to Churn"
else:
    prediction_sent = "The customer is not likely to Churn"

# Display prediction
st.write("Prediction:", prediction_sent)