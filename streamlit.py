import streamlit as st
import pandas as pd

from src.ml_project.pipelines.prediction_pipeline import PredictPipeline

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("Predict **Math Score** using ML model")

# Input Fields
gender = st.selectbox("Gender", ["male", "female"])

race_ethnicity = st.selectbox(
    "Race / Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ]
)

lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

test_preparation_course = st.selectbox(
    "Test Preparation Course", ["none", "completed"]
)

reading_score = st.number_input("Reading Score", 0, 100, step=1)
writing_score = st.number_input("Writing Score", 0, 100, step=1)

# Predict Button
if st.button("Predict Math Score"):
    data = pd.DataFrame({
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_level_of_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score],
    })

    pipeline = PredictPipeline()
    prediction = pipeline.predict(data)

    st.success(f"📊 Predicted Math Score: **{round(prediction[0], 2)}**")
