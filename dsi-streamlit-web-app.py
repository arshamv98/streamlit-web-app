import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.joblib")

st.title("Purchase Prediction Model")
st.subheader("Enter Cust. Info")

age=st.number_input("Age",min_value=18,max_value=120,value=18)
gender = st.selectbox("Gender", ["M", "F"])
credit_score = st.number_input("Credit_Score",max_value=1000,value=500)
if st.button("Submit"):
    new_data = pd.DataFrame({"age":[age],"gender":[gender],"credit_score":[credit_score]})
    pred_proba = model.predict_proba(new_data)[0][1]
    st.subheader(f"{pred_proba:.0%}")



