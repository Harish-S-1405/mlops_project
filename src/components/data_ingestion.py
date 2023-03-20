import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import argparse
from src.utils import read_yaml,bucket
import json

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig 

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str

class DataIngestion:
    def __init__(self,a,b,c):
        self.ingestion_config = DataIngestionConfig(
            train_data_path=os.path.join('artifacts', a),
            test_data_path=os.path.join('artifacts', b),
            raw_data_path=os.path.join('artifacts', c)
        )

    def initiate_data_ingestion(self,params_path):

        params=read_yaml(params_path)

        print(self.ingestion_config.train_data_path)
        print(self.ingestion_config.test_data_path)


        logging.info('Enter the data ingestion method or component')
        try:


            ## Local upload
            
            data_path=params['Data_upload']['upload_from_local']['path']
            df=pd.read_csv(data_path)


            # ###  GCP

            # with open(params['Data_upload']["upload_from_gcp"]['json_file_path']) as f:
            #       json_file = json.load(f)
            # df = bucket(json_file,params['Data_upload']["upload_from_gcp"]['bucket_name'],params['Data_upload']["upload_from_gcp"]['file_path_name'],params['Data_upload']["upload_from_gcp"]['cloud_name'])


            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
        
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train test split initiated')

            split_ratio=params['base']['test_split_ratio']
            random_state=params['base']['random_state']

            train_set,test_set=train_test_split(df,test_size=split_ratio,random_state=random_state)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':

    args=argparse.ArgumentParser()
    args.add_argument("--params","-p",default="params.yaml")

    parsed_args=args.parse_args()

    params = read_yaml(parsed_args.params)
    a = params['Data_upload']['upload_from_local']['a']
    b = params['Data_upload']['upload_from_local']['b']
    c = params['Data_upload']['upload_from_local']['c']


    obj=DataIngestion(a,b,c)
    train_data_path,test_data_path= obj.initiate_data_ingestion(params_path=parsed_args.params)   


    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path,params_path=parsed_args.params)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,params_path=parsed_args.params))