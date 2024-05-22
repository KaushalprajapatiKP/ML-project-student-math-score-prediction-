import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    This class is used to store all the configuration for data transformation
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        """
        This function will return the data transformation object
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", 
                "race_ethnicity", 
                "parental_level_of_education", 
                "lunch",
                "test_preparation_course"
                ]
            
            num_pipeline = Pipeline (
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std_scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Caltegorical columns : {categorical_columns}")
            logging.info(f"Numerical columns : {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function will initiate the data transformation
        """
        try:
            logging.info("Initiating data transformation")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Reading of training and test data completed.")
            logging.info(f"Train data shape : {train_data.shape}")
            logging.info(f"Test data shape : {test_data.shape}")
            logging.info("Obtaining preprocessing object...")

            preprocessor = self.get_data_transformation_obj()
            logging.info("Obtaining preprocessing object completed.")
            logging.info("Transforming training data...")
            target_column_name = "math_score"
            numerical_columns = ["writing_data", "reading_data"]

            input_feature_train_df = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_data[target_column_name]

            input_feature_test_df = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_data[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe...")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                object= preprocessor
            )

            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)