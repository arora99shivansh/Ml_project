import sys
import pandas as pd
from src.ml_project.utils import load_object
from src.ml_project.exception import CustomException


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model = load_object("artifacts/model.pkl")
            preprocessor = load_object("artifacts/preprocessor.pkl")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
