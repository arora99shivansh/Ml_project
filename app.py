import sys
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException

from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_transformation import DataTransformation
from src.ml_project.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    logging.info("Execution started")

    try:
        # 1️⃣ Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        logging.info(
            f"Data ingestion completed | "
            f"Train file: {train_data_path}, Test file: {test_data_path}"
        )

        # 2️⃣ Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        logging.info("Data transformation completed")

        # 3️⃣ Model Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model training completed with R2 score: {r2_score}")

    except Exception as e:
        logging.error("Custom Exception occurred", exc_info=True)
        raise CustomException(e, sys)
