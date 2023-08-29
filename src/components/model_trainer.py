import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustopException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class Modeel_trainer_config:
    trained_model_file_path=os.path.join('artifact','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=Modeel_trainer_config()
    
    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            #logging("Spliting Training and test input data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            moodels={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Linear Regression':LinearRegression(),
                'k-Neighbours Regressor':KNeighborsRegressor(),
                'CatBoosting Regressior':CatBoostRegressor(verbose=False),
                'AdaBoost Regressior':AdaBoostRegressor()


                }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=moodels)

            best_model_score=max(model_report.values())
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=moodels[best_model_name]

            if best_model_score<0.6:
                raise CustopException("No best Model found")
            logging.info(f"Best found model on both training and testing dataet")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_squqre=r2_score(y_test,predicted)

            return r2_squqre

        except Exception as e:
            raise CustopException(e,sys)
        

            