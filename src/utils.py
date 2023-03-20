import yaml
import os
import json
import pandas as pd
import pyodbc 
import mysql.connector as msc
import io
from io import BytesIO
from google.cloud import storage
import boto3
import json
import yaml
import numpy as np
import sys
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from scipy.stats import randint as sp_randint
from sklearn.ensemble import (
     RandomForestRegressor,
     AdaBoostRegressor,
     GradientBoostingRegressor)

def bucket(credentials_dict, bucket_name, file_name_path_or_object_key, cloud_name):
    if cloud_name.lower()=="gcp":
        storage_client = storage.Client.from_service_account_info(credentials_dict)
        BUCKET_NAME = bucket_name
        bucket = storage_client.get_bucket(BUCKET_NAME)
        filename = list(bucket.list_blobs(prefix=''))
        for name in filename:
            print(name.name)
        blob = bucket.blob(file_name_path_or_object_key)
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data), encoding="utf-8", sep=",")
        return df


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content

def save_object(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(true, predicted): 

    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)

    return mae, rmse, r2_square

def train_models(X_train,y_train,X_test,y_test,models):

    try:
        ## train_report={}
        ## test_report={}

        report1={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_model_score1 = r2_score(y_train, y_train_pred)

            test_model_score1 = r2_score(y_test, y_test_pred)

            ## model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
            ## model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

            ## metrics=['rmse','mae','r2']
            ## values_train=[model_train_rmse,model_train_mae,model_train_r2]
            ## values_test=[model_test_rmse,model_test_mae,model_test_r2]

            report1[list(models.keys())[i]] = (test_model_score1,model)

            ## final1=dict(zip(metrics,values_train))
            ## final2=dict(zip(metrics,values_test))

            ## train_report[list(models.keys())[i]]=final1
            ## test_report[list(models.keys())[i]]=final2


        return report1
    
    except Exception as e:
        raise CustomException(e,sys)
    

def tuning(X_train,y_train,X_test,y_test,tune_models,params_path):

    params=read_yaml(params_path)
    param=dict(list(params.items())[2:])
    print(param)
    try:
        ## train_report1={}
        ## test_report1={}

        report2={}


        for i in range(len(list(tune_models))):
            model=list(tune_models.values())[i]
            algo=list(tune_models.keys())[i]
            random_grid=param[list(tune_models.keys())[i]]
            print(random_grid)
            # if(algo=="AdaBoost_Regressor_Tuned"):
            #     random_grid['estimator']=[RandomForestRegressor()]


            model_tuning = RandomizedSearchCV(estimator = model, param_distributions = random_grid,
                        n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
            model_tuning.fit(X_train, y_train)

            best_estimator = model_tuning.best_estimator_

            y_train_pred=best_estimator.predict(X_train)
            y_test_pred=best_estimator.predict(X_test)

            train_model_score2 = r2_score(y_train, y_train_pred)

            test_model_score2 = r2_score(y_test, y_test_pred)

            ## model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
            ## model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

            ## metrics=['rmse','mae','r2']
            ## values_train=[model_train_rmse,model_train_mae,model_train_r2]
            ## values_test=[model_test_rmse,model_test_mae,model_test_r2]

            report2[list(tune_models.keys())[i]] = (test_model_score2,best_estimator)

            ## final1=dict(zip(metrics,values_train))
            ## final2=dict(zip(metrics,values_test))

            ## train_report1[list(tune_models.keys())[i]]=final1
            ## test_report1[list(tune_models.keys())[i]]=final2
    
        return report2

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
