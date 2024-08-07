import os
import sys

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj: #wb- write byte
            dill.dump(obj,file_obj)
            # dill- library which will help us to  create the pickle file
    except Exception as e:
        raise CustomException(e,sys)
        

def evaluate_model(xtrain,ytrain,xtest,ytest,models):
    try:
        report={}


        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(xtrain,ytrain)

            ytrain_pred=model.predict(xtrain)

            ytest_pred=model.predict(xtest)

            train_model_score=r2_score(ytrain,ytrain_pred)

            test_model_score=r2_score(ytest,ytest_pred)

            report[list(models.keys())[i]]=test_model_score

        return report

    except:
        pass