import os
import sys

from src.logger import logging 
from src.exceptions import CustomException
from src.utils import save_object

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np 

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Started!")
            # Now define which columns should be ordinal encoded and which should be scaled
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Now, define the custom ranking for each ordinal Variable

            cut_categories = ['Fair', "Good", 'Very Good', 'Premium', 'Ideal']
            color_categories = ["J", "I", "H", "G", "F", "E", "D"]
            clarity_categories = ['I1', "SI1", "SI2", 'VS1', 'VS2', 'VVS2', 'VVS1', 'IF']
            
            ## Numerical Pipeline 
            logging.info("pipeline Initiated!")
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('scaler', StandardScaler())       
                ]
            )
            
            # Combining 2 pipelines into 1 
            preprocessor = ColumnTransformer([
                            ('num_pipeline', num_pipeline, numerical_cols),
                            ('cat_pipeline', cat_pipeline, categorical_cols)
                            ])
            logging.info("Pipeline Completed Successfully!")

            return preprocessor
        except Exception as e:
            logging.info('Error in Data Transformation Stage!')
            raise CustomException(e, sys)
    

    def initiate_data_transformation(self, train_path, test_path):
        try:

            # Reading Train and Test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Of Data Initiated!")
            logging.info(f"Train Data: \n {train_df.head().to_string()}")
            logging.info(f"Test Data: \n {test_df.head().to_string()}")

            logging.info("Data read Successfully!")

            preprocessor_obj = self.get_data_transformation_object()

            target_col_name = "price"
            drop_column = [target_col_name, 'id']

            # Creating Separate independent and Dependent Features for Train and Test data
            input_feature_train_df = train_df.drop(columns= drop_column, axis=1)
            target_feature_train_df = train_df[target_col_name]

            input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[target_col_name]

            # Transforming using Preprocessor Object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Now, Let's save the train newly preprocessed data in array form just in case. 
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #np.c_ is used to concatenate two arrays.
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )
            logging.info("Pickle file Created")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_Data_Transformation function!")
            raise CustomException(e, sys)