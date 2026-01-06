from flask import Flask, render_template, request
import pandas as pd

from src.ml_project.pipelines.prediction_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """
    Home page – form render karega
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Form se data lega → DataFrame banayega → PredictPipeline call karega
    """

    try:
        # 🔹 Form data collect
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get(
            "parental_level_of_education"
        )
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get(
            "test_preparation_course"
        )
        reading_score = float(request.form.get("reading_score"))
        writing_score = float(request.form.get("writing_score"))

        # 🔹 DataFrame exactly training jaisa
        input_data = pd.DataFrame({
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [parental_level_of_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation_course],
            "reading_score": [reading_score],
            "writing_score": [writing_score]
        })

        # 🔹 Prediction
        pipeline = PredictPipeline()
        result = pipeline.predict(input_data)

        prediction = round(float(result[0]), 2)

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Error occurred"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
