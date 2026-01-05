import os
import sys
import pandas as pd

from src.ml_project.exception import CustomException
from src.ml_project.utils import load_object


class PredictPipeline:
    def predict(self, features):
        try:
            ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

            model_path = os.path.join(ROOT_DIR, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(ROOT_DIR, "artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)

        except Exception as e:
            raise CustomException(e, sys)
