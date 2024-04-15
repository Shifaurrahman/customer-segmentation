import streamlit as st
import numpy as np
import pickle

with open('website/predictor.pickle', 'rb') as file:
    kmeans = pickle.load(file)

def main():
    st.title('Customer Segmentation Predictor')

    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
    annual_income = st.number_input('Annual Income (k$)', min_value=0, max_value=150, value=50, step=1)
    spending_score = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50, step=1)

    gender_encoded = 0 if gender == 'Male' else 1

    new_data = np.array([[age, annual_income, spending_score]])
    predicted_cluster = kmeans.predict(new_data)

    if st.button("Predict Label"):
        st.success(f"Predicted Cluster Label: {predicted_cluster[0]}")

if __name__ == '__main__':
    main()
