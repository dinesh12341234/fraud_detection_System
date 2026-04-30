import streamlit as st
import joblib
import pandas as pd
import lightgbm as lgb
from geopy.distance import geodesic

model = joblib.load('fraud_detection_model.jb')
encoders = joblib.load('label_encoders.jb')

def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.title("Fraud Detection system")
st.write("Enter the transaction details below")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0)
lat = st.number_input("Latitude", format="%.6f")
long = st.number_input("Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.number_input("Transaction Hour", 0,23,12)
day = st.number_input("Transaction Day", 1,31,15)
month = st.number_input("Transaction Month", 1,12,6)
gender = st.selectbox("gender", ["M", "F"])
cc_num = st.text_input("Credit Card Number")

distance = haversine_distance(lat, long, merch_lat, merch_long)


if st.button("Check for Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month , gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month','gender','cc_num'])
        
        categorical_cols = ['merchant', 'category', 'gender']
        for col in categorical_cols:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
        prediction = model.predict(input_data)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        st.subheader(f"Transaction is: {result}")
    else:
        st.error("Please Fill in all required fields (Merchant, Category, Credit Card Number)")
    
