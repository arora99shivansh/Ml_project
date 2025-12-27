import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor


# ==============================
# Configuration class
# ==============================
@dataclass
class ModelTrainerConfig:
    """
    Stores path where trained model will be saved
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


# ==============================
# Model Trainer class
# ==============================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple ML models, selects best model based on R2 score,
        and saves the best model.
        """
        try:
            logging.info("Splitting training and testing data")

            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define models with stable hyperparameters
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False,
                    random_state=42
                ),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            model_report = {}          # Stores R2 score of each model
            trained_models = {}        # Stores trained model objects

            logging.info("Training models and evaluating performance")

            # Train & evaluate each model
            for model_name, model in models.items():
                model.fit(X_train, y_train)              # Train model
                y_pred = model.predict(X_test)           # Predict
                r2 = r2_score(y_test, y_pred)             # Evaluate

                model_report[model_name] = r2
                trained_models[model_name] = model

                logging.info(f"{model_name} R2 score: {r2}")

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No suitable model found with R2 score >= 0.6",
                    sys
                )

            logging.info(
                f"Best model selected: {best_model_name} "
                f"with R2 score: {best_model_score}"
            )

            # Save the best trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Return final R2 score
            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
