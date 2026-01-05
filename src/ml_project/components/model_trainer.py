import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
from src.ml_project.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                },
                "Random Forest": {
                    "n_estimators": [64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                },
                "Linear Regression": {},
                "XGBoost": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                },
                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                },
            }

            logging.info("Evaluating models")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best R2 Score: {best_model_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            print("\n==============================")
            print(f"✅ Best Model     : {best_model_name}")
            print(f"✅ R2 Score       : {best_model_score}")
            print("==============================\n")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
