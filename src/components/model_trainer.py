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

from sklearn.model_selection import GridSearchCV


# ==============================
# Configuration class
# ==============================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


# ==============================
# Model Trainer class
# ==============================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple ML models using GridSearchCV,
        selects best model based on R2 score and saves it.
        """
        try:
            logging.info("Splitting training and testing data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Models
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            # Hyperparameters
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse']
                },
                "Random Forest": {
                    'n_estimators': [16, 32, 64, 128]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [16, 32, 64]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7]
                },
                "XGB Regressor": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [32, 64]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [32, 64]
                }
            }

            best_model_score = -1
            best_model = None
            best_model_name = None

            logging.info("Starting model training with hyperparameter tuning")

            # GridSearchCV for each model
            for model_name, model in models.items():
                logging.info(f"Tuning model: {model_name}")

                param_grid = params[model_name]

                if len(param_grid) == 0:
                    # No params → directly train
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                else:
                    gs = GridSearchCV(
                        model,
                        param_grid,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1
                    )
                    gs.fit(X_train, y_train)

                    model = gs.best_estimator_
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)

                logging.info(f"{model_name} R2 score: {r2}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model = model
                    best_model_name = model_name

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with R2 score >= 0.6",
                    sys
                )

            logging.info(
                f"Best Model: {best_model_name} "
                f"with R2 Score: {best_model_score}"
            )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
