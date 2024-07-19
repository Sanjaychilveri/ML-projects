import os
import sys
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
from xgboost import XGBRegressor


from src.logger import logging
from src.exception import CustomException

  
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_arr,test_arr):
        try:

            logging.info("Split training and test input data")

            xtrain,ytrain,xtest,ytest=[
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            ]
            
            models={

                "random forest":RandomForestRegressor(),
                'catboosting classifier':CatBoostRegressor(verbose=False),
                'Decision tree': DecisionTreeRegressor(),
                'Gradient boosting': GradientBoostingRegressor(),
                'linear regression': LinearRegression(),
                'K-neighbours classifier':KNeighborsRegressor(),
                'adaboost classifier':AdaBoostRegressor(),
                'XGBclassifier':XGBRegressor(),

            }            

            model_report:dict=evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,
                                             ytest=ytest,models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info('Best model found on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(xtest)

            r2_square=r2_score(ytest,predicted)
            return (r2_square,best_model_name)

        except Exception as e:
            raise CustomException(e,sys)
