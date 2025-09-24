import streamlit as st
import requests

st.title("Minimal model demo")
st.write("Enter features for prediction:")

# Feature names
feature_names = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6","target"]
inputs = {}

# Create number inputs for each feature
for name in feature_names:
    inputs[name] = st.number_input(name, value=0.0)

if st.button("Predict"):
    payload = inputs  # Send as dict to match FastAPI PredictRequest
    try:
        resp = requests.post("http://api:8000/predict", json=payload, timeout=5)
        st.write(resp.json())
    except Exception as e:
        st.error(f"Request failed: {e}")
