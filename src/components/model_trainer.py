import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "KNN": KNeighborsRegressor(),
            }

            model_report:dict= evaluate_models(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            print("\nModel Performance Report:")
            for name, score in model_report.items():
              print(f"{name}: {score:.4f}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found", sys)

            logging.info(f"Best model: {best_model_name} with R2 score {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            print("\nSample Actual vs Predicted Values:")
            for actual, prediction in zip(y_test[:10], predicted[:10]):
             print(f"Actual: {actual:.2f} | Predicted: {prediction:.2f}")
            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = mean_squared_error(y_test, predicted) ** 0.5

            print("\nFinal Best Model Evaluation:")
            print(f"Best Model: {best_model_name}")
            print(f"R² Score: {r2_square:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
