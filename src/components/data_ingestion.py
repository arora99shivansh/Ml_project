# ============================
# DATA INGESTION COMPONENT
# ============================

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.exception import CustomException
from src.logger import logging


# -------------------------------------------------
# Configuration class to store file paths
# -------------------------------------------------
# Dataclass is used to avoid hardcoding paths
# and to keep configuration clean and manageable
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


# -------------------------------------------------
# Data Ingestion class
# -------------------------------------------------
# This class is responsible for:
# 1. Reading raw data
# 2. Saving raw data
# 3. Splitting data into train and test
# 4. Saving train & test data
class DataIngestion:
    def __init__(self):
        # Initialize ingestion configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method")

        try:
            # ---------------------------------
            # Step 1: Read the dataset
            # ---------------------------------
            # Using os.path.join for OS-independent path handling
            df = pd.read_csv(
                os.path.join('notebook', 'data', 'stud.csv')
            )
            logging.info("Dataset read successfully into dataframe")

            # ---------------------------------
            # Step 2: Create artifacts directory
            # ---------------------------------
            # os.path.dirname is used to extract folder path
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )
            logging.info("Artifacts directory created")

            # ---------------------------------
            # Step 3: Save raw data
            # ---------------------------------
            # Raw data is saved for future reference/debugging
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )
            logging.info("Raw data saved successfully")

            # ---------------------------------
            # Step 4: Train-Test Split
            # ---------------------------------
            # Splitting data into 80% training and 20% testing
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # ---------------------------------
            # Step 5: Save train data
            # ---------------------------------
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # ---------------------------------
            # Step 6: Save test data
            # ---------------------------------
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            # ---------------------------------
            # Step 7: Return file paths
            # ---------------------------------
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            # Log error and raise custom exception
            logging.error("Error occurred during Data Ingestion")
            raise CustomException(e, sys)


if __name__ =="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_tranformation=DataTransformation()
    data_tranformation.initiate_data_transform(train_data,test_data)