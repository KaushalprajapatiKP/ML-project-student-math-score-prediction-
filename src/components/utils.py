import os, sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            dill.dump(object, file_object)

    except Exception as e:
        raise CustomException(e, error_details=e)   