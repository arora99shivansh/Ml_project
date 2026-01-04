import sys
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    logging.info("Execution started")

    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        logging.info(f"Data ingestion completed. "
                     f"Train file: {train_path}, Test file: {test_path}")

    except Exception as e:
        logging.error("Custom Exception occurred", exc_info=True)
        raise CustomException(e, sys)
