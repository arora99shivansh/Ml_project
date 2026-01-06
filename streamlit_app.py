import streamlit as st
import pandas as pd

from src.ml_project.pipelines.prediction_pipeline import PredictPipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🎓 Student Performance Predictor</h1>
    <p style="text-align:center; color:gray;">
        Predict <b>Math Score</b> based on student details using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("📌 About Project")
st.sidebar.write(
    """
    - End-to-End ML Project  
    - Data Ingestion → Transformation → Training  
    - Models compared automatically  
    - Best model selected using R² score  
    """
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Built by **Shivansh Arora**")

# ---------------- INPUT FORM ----------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["male", "female"])
        race_ethnicity = st.selectbox(
            "Race / Ethnicity",
            ["group A", "group B", "group C", "group D", "group E"]
        )
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

    with col2:
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
        test_preparation_course = st.selectbox(
            "Test Preparation Course", ["none", "completed"]
        )

    st.markdown("### 📊 Academic Scores")
    col3, col4 = st.columns(2)

    with col3:
        reading_score = st.number_input(
            "Reading Score", min_value=0, max_value=100, value=50
        )

    with col4:
        writing_score = st.number_input(
            "Writing Score", min_value=0, max_value=100, value=50
        )

    submit = st.form_submit_button("🚀 Predict Math Score")

# ---------------- PREDICTION ----------------
if submit:
    try:
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
        result = pipeline.predict(data)

        st.success("✅ Prediction Successful")

        st.markdown(
            f"""
            <div style="
                background-color:#f0f2f6;
                padding:20px;
                border-radius:10px;
                text-align:center;
                font-size:22px;
            ">
                📈 <b>Predicted Math Score</b><br><br>
                <span style="font-size:32px; color:#1f77b4;">
                    {round(result[0], 2)}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Something went wrong while predicting.")

