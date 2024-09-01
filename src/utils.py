import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj: #wb- write byte
            dill.dump(obj,file_obj)
            # dill- library which will help us to  create the pickle file
    except Exception as e:
        raise CustomException(e,sys)
        

def evaluate_models(xtrain,ytrain,xtest,ytest,models,params):
    try:
        report={}


        for i in range(len(list(models))):
            model=list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(xtrain,ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain,ytrain)

            ytrain_pred=model.predict(xtrain)

            ytest_pred=model.predict(xtest)

            train_model_score=r2_score(ytrain,ytrain_pred)

            test_model_score=r2_score(ytest,ytest_pred)

            report[list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    
    