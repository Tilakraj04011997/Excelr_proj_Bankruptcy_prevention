# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:59:13 2024
Author: rajti
Model: Support Vector Machine (SVM)
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained SVM model
load_model = pickle.load(open('C:/Users/rajti/Desktop/Excelr project/trained_model_new1.save', 'rb'))

def bankruptcy_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = load_model.predict(input_data_as_reshaped)
    return "bankruptcy" if prediction[0] == 0 else "non-bankruptcy"

def main():
    st.title('Bankruptcy Prediction Web App')
    st.title('Model - Support Vector Machine')

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Input fields and labels in left column
    with col1:
        st.write("Input Features:")
        industrial_risk = st.text_input('Industrial Risk')
        management_risk = st.text_input('Management Risk')
        financial_flexibility = st.text_input('Financial Flexibility')
        credibility = st.text_input('Credibility')
        competitiveness = st.text_input('Competitiveness')
        operating_risk = st.text_input('Operating Risk')

    # Prediction result label in right column
    with col2:
        if st.button('Output result - Predict Bankruptcy'):
            banking = bankruptcy_prediction([industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk])
            st.write(f"Prediction: {banking}")

            
             

if __name__ == '__main__':
    main()
