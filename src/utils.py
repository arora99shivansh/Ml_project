import os
import sys
import dill # Better than pickle for complex ML objects

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object (model, preprocessor, pipeline)
    using dill serialization.
    """
    try:
        # Extract directory from file path
        dir_path = os.path.dirname(file_path)

        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)

        # Save object in binary format
        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)
