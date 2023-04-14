import os
import sys
from src.logger import logging 
from src.exceptions import CustomException
from src.components.data_transformation import DataTransformation
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## Initialize Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

## Create a class for Data Ingestion

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Methods Starts")
        try:
            df = pd.read_csv('notebooks/data/gemstone.csv') # To reach the parent path we have to add ./
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Train Test Split")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=30)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header = True)

            
            logging.info("Data Ingestion Is Completed!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception Occured at Data Ingestion Stage!")
            raise CustomException(e, sys)
        

