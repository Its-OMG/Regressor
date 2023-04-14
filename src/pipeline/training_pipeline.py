import os
import sys
from src.logger import logging 
from src.exceptions import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
import pandas as pd







## run Data Ingestion

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    model_train = ModelTrainer()
    model_train.initiate_model_training(train_arr, test_arr)
