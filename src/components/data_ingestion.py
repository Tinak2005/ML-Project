import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

class DataIngestion:
    def __init__(self):
        logging.info("Data Ingestion class initialized")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")

            # Example: read CSV file (replace with your dataset path)
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Dataset loaded successfully")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed")

            # Save train and test data
            os.makedirs("artifacts", exist_ok=True)
            train_set.to_csv(os.path.join("artifacts", "train.csv"), index=False, header=True)
            test_set.to_csv(os.path.join("artifacts", "test.csv"), index=False, header=True)

            logging.info("Data ingestion completed successfully")
            return (
                os.path.join("artifacts", "train.csv"),
                os.path.join("artifacts", "test.csv")
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__=="__main__":
       obj=DataIngestion()
       train_data, test_data =obj.initiate_data_ingestion()

       data_transformation=DataTransformation()
       train_arr, test_arr,preprocessor_path=data_transformation.initiate_data_transformation(train_data, test_data)

       model_trainer=ModelTrainer()
       print(model_trainer.initiate_model_trainer(train_arr, test_arr))
