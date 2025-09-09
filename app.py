import streamlit as st
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()

st.title("XOR Prediction App")

x1 = st.number_input("Enter X1 (0 or 1)", min_value=0, max_value=1, step=1)
x2 = st.number_input("Enter X2 (0 or 1)", min_value=0, max_value=1, step=1)

if st.button("Predict"):
    pred = model.predict(np.array([[x1, x2]]))[0][0]
    st.write(f"Prediction: {round(pred)} (Probability: {pred:.4f})")


# STEPS:
# 1. git init
# 2. make train.py and app.py
# 3. git add .
# 4. git commiy -m "some message"
# 5. git branch -M main
# 6. git remote add origin https://github.com/riddhiiee/dummy-projj.git 
# 7. git push -u origin main
# 8. git push -u origin main --force