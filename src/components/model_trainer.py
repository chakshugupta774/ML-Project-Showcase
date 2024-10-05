import sys
import os  
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_fit_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing input data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models = {
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regression" : KNeighborsRegressor(),
                "Decision Tree"  : DecisionTreeRegressor(),
                "random forest Regressor" : RandomForestRegressor(),
                "XGBregressor" : XGBRegressor(),
                "CatBossting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor()
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models = models)

            #tot get beset model score from dict 
            best_model_score = max(sorted(model_report.values()))

            #to get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found !!")
            logging.info("Best logging found on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_fit_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square 
        except Exception as e:
            raise CustomException(e,sys)