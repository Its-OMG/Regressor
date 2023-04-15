import os 
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import pickle

from src.exceptions import CustomException
from src.logger import logging


# Function to save Model as Pickle File
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



# Function for Evaluating the Models One by One
def evaluate_model(X_train, y_train, X_test, y_test, models):
    
    report = {}
    for i in range(len(models)):
        model = list(models.values())[i]

        #Train Each Model 
        model.fit(X_train, y_train)

        #Make Predictions
        y_test_pred = model.predict(X_test)

        #Test model score
        test_model_score = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]] = test_model_score

    return report



# Function For loading a pickle File
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info("Error Occred While loading Pickle File")
        raise CustomException(e, sys)