# feature engineering + EDA
# This file is responsible for feature engineering and data preprocessing
# It prepares raw data so that it can be fed into machine learning models

import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object   # utility to save pickle file


# Configuration class
@dataclass
class DataTransformationConfig:
    # Path where the trained preprocessing object will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        """
        Creates preprocessing pipelines for numerical and categorical data
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline: handle missing values + scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline: imputation + encoding + scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transform(self, train_path, test_path):
        """
        Reads train/test data, applies preprocessing,
        and returns transformed arrays
        """
        try:
            # -----------------------------
            # Step 1: Read datasets
            # -----------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")

            # -----------------------------
            # Step 2: Get preprocessing object
            # -----------------------------
            preprocessing_obj = self.get_data_transform_object()

            target_column_name = "math_score"

            # -----------------------------
            # Step 3: Split input and target features
            # -----------------------------
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing data")

            # -----------------------------
            # Step 4: Apply transformations
            # -----------------------------
            # Fit only on training data to avoid data leakage
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Only transform test data
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # -----------------------------
            # Step 5: Combine input features with target
            # -----------------------------
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            # -----------------------------
            # Step 6: Save preprocessing object
            # -----------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            return train_arr, test_arr

        except Exception as e:
            logging.error("Error in initiate_data_transform")
            raise CustomException(e, sys)
