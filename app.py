import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import load_model
from prediction import get_prediction, ordinal_encoder

model = load_model('keras_model.h5')

#page settings
st.set_page_config(page_title="Patient Death Prediction App",
                   page_icon="⚕️", layout="wide")

#page header
st.title(f'Patient Death Predictor App')

#Creating option list for dropdown menu
gender_options = ['M','F']
admit_options = ['Floor','Accident & Emergency','Operating Rom/Recovery','Other Hospital','Other ICU']
diabetes_options = [0,1]
cirrhosis_options = [0,1]
aids_options = [0,1]
hepatic_options = [0,1]
immuno_options = [0,1]
leuk_options = [0,1]
lymph_options = [0,1]
tumor_options = [0,1]

features = ['gender','age','height','weight','icu_admit_source','heart_rate_apache','glucose_apache','diabetes_mellitus','cirrhosis','aids','hepatic_failure','immunosuppression','leukamia','lymphoma','solid_tumor_with_metastasis']
st.markdown("<h1 style='text-align:center;'>Patient Death Predictor App⚕️</h1>",unsafe_allow_html=True)

def main():
    with st.form('prediction_form'):
        st.subheader("Enter the input for following features:")
        
        gender = st.selectbox("Select Gender:",options=gender_options)
        age = st.slider("Pickup Age:",0,100,value=0,format="%d")
        height = st.slider("Pickup Height in cm:",0,200,value=0,format="%d")
        weight = st.slider("Pickup Weight in kg:",0,200,value=0,format="%d")
        icu_admit_source = st.selectbox("Select ICU Admit Source:",options=admit_options)
        heart_rate_apache = st.slider("Pickup Heart Rate:",0,180,value=0,format="%d")
        glucose_apache = st.slider("Pickup Glucose Level:",0,600,value=0,format="%d")
        diabetes_mellitus = st.selectbox("Select If Diabetes (0-'No', 1-'Yes):",options=diabetes_options)
        cirrhosis = st.selectbox("Select If Cirrhosis (0-'No', 1-'Yes):",options=cirrhosis_options)
        aids = st.selectbox("Select If Aids (0-'No', 1-'Yes):",options=aids_options)
        hepatic_failure = st.selectbox("Select If Hepatic Failure (0-'No', 1-'Yes):",options=hepatic_options)
        immunosuppression = st.selectbox("Select If Immunosuppression (0-'No', 1-'Yes):",options=immuno_options)
        leukamia = st.selectbox("Select If Leukamia (0-'No', 1-'Yes):",options=leuk_options)
        lymphoma = st.selectbox("Select If Lymphoma (0-'No', 1-'Yes):",options=lymph_options)
        solid_tumor_with_metastasis = st.selectbox("Select If Tumor (0-'No', 1-'Yes):",options=tumor_options)
        
        submit = st.form_submit_button("Predict")

    if submit:
        gender = ordinal_encoder(gender, gender_options)
        icu_admit_source = ordinal_encoder(icu_admit_source, admit_options)
        
        data = np.array([gender,age,height,weight,icu_admit_source,heart_rate_apache,glucose_apache,diabetes_mellitus,cirrhosis,aids,hepatic_failure,immunosuppression,leukamia,lymphoma,solid_tumor_with_metastasis]).reshape(1,-1)
        pred = get_prediction(data=Data, model=model)

        st.write(f"The Patient is (0 for Dead/ 1 for Survive):  {pred[0]}")

if __name__ == '__main__':
    main()