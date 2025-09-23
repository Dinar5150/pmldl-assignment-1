import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

st.sidebar.header("Input Features")

features = {}
for name in ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]:
    features[name] = st.sidebar.number_input(name, value=0.0, format="%.6f")

if st.button("Predict"):
    payload = {f: features[f] for f in ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]}
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            pred = data.get("prediction")
            st.success(f"Prediction: {pred:.2f}")
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
