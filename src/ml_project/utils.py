import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.ml_project.exception import CustomException
from src.ml_project.logger import logging

load_dotenv()

host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
db = os.getenv("MYSQL_DATABASE")


def read_sql_data():
    logging.info("Reading data from MySQL database using SQLAlchemy")

    try:
        engine = create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        )

        df = pd.read_sql_query("SELECT * FROM student", engine)

        logging.info("Data fetched successfully from student table")
        return df

    except Exception as e:
        logging.error("Error while reading SQL data")
        raise CustomException(e, sys)
