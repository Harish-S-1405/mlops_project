# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
from dataclasses import dataclass

# Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
     RandomForestRegressor,
     AdaBoostRegressor,
     GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,read_yaml,train_models,tuning

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array,params_path):

        

        try:
            logging.info("split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            base_models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            tun_models= {
                'lasso_tuned': Lasso(),
                'Ridge_tuned': Ridge(),
                'K_Neighbors_Regressor_tuned':KNeighborsRegressor(),
                "Decision_Tree_Tuned": DecisionTreeRegressor(),
                "Random_Forest_Regressor_Tuned": RandomForestRegressor(),
                "XGBRegressor_Tuned": XGBRegressor(),
                "CatBoosting_Regressor_Tuned": CatBoostRegressor(verbose=False),
                "AdaBoost_Regressor_Tuned": AdaBoostRegressor()
            }

            model_report1=train_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=base_models)
            model_report2=tuning(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,tune_models=tun_models,params_path=params_path)

            model_report = model_report1.copy()
            model_report.update(model_report2)


            values_dict=list(model_report.values())
            best_model_score = max([i[0] for i in values_dict])
            # print(f"Best Model Score: {best_model_score}")
            best_model = [i[1] for i in model_report.values() if i[0]==max([i[0] for i in model_report.values()])][0]

            # train_final1=pd.DataFrame(base_train_report).T
            # test_final1=pd.DataFrame(base_test_report).T

            # train_final2=pd.DataFrame(tune_train_report).T
            # test_final2=pd.DataFrame(tune_test_report).T

            # print(train_report)
            # print(test_report)

            # final_train_report=pd.concat([train_final1,train_final2],axis=0)
            # final_test_report=pd.concat([test_final1,test_final2],axis=0)

            # test_report = base_test_report.copy()
            # test_report.update(tune_test_report)

            # r2s=[]
            # for i in test_report.values():
            #     r2s.append(i['r2'])

            # report=dict(zip(test_report.keys(),r2s))

            # best_model_score=max(sorted(report.values()))

    
            # best_model_name=list(report.keys())[
            #     list(report.values()).index(best_model_score)
            # ]

            # try:
            #     best_model=base_models[best_model_name]
            # except:
            #     pass

            # try:
            #     best_model=tun_models[best_model_name]
            # except:
            #     pass

            # print(best_model)

            if best_model_score<0.6:
                raise CustomException('no best model found')
            
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_sq=r2_score(y_test,predicted)
            return r2_sq

        except Exception as e:
            raise CustomException(e,sys)
            




