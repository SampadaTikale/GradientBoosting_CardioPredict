import pickle
import streamlit as st
import numpy as np
import cv2
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open(r'E:\Engineer1\6sem\ML_Project\My_Project\classifier.pkl', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Cardiovascular Disease Prediction System',
                           ['Cardiovascular Disease Prediction'],
                           icons=['heart'],
                           default_index=0)

# Cardiovascular Disease Prediction Page
if selected == 'Cardiovascular Disease Prediction':
    # page title
    st.title('Cardiovascular Disease Prediction Using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        id = st.text_input('Patient ID Number')

    with col2:
        age = st.text_input('Patient Age')

    with col3:
        gender = st.text_input('Patient Gender')

    with col1:
        height = st.text_input('Patient Height')

    with col2:
        weight = st.text_input('Patient Weight')

    with col3:
        ap_hi = st.text_input('Patient Systolic Blood Pressure')

    with col1:
        ap_lo = st.text_input('Patient Diastolic Blood Pressure')

    with col2:
        cholesterol = st.text_input('Patient Cholesterol')

    with col3:
        gluc = st.text_input('Patient Glucose')

    with col1:
        smoke = st.text_input('Patient Is Smoking')

    with col2:
        alco = st.text_input('Patient Is Alocoholic')

    with col3:
        active = st.text_input('Patient Is Physical Active')

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    diab_diagnosis = ''

    if st.button('Cardiovascular Disease Prediction Test Result'):
        # Process the uploaded image
        if uploaded_image is not None:
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            # Perform image processing and data extraction here
            # Extracted data can be used in place of the user input fields above
            # For example: age = extracted_age, gender = extracted_gender, etc.
            
            # Now make the prediction using the extracted data
            diab_prediction = diabetes_model.predict([[id, age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The Person has CVD (Cardiovascular Disease)'
            else:
                diab_diagnosis = 'The Person does not have CVD (Cardiovascular Disease Prediction)'

    st.success(diab_diagnosis)
