import os
import sys
import dill   # pickle ki jagah dill use kiya gaya hai
from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
from sklearn.metrics import r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates multiple models, returning a report of R² scores.
    
    Parameters:
        X_train, y_train: Training features and target
        X_test, y_test: Testing features and target
        models: Dictionary of model_name -> model_instance
    
    Returns:
        dict: model_name -> R² score
    """
    report = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Evaluate using R² score
        score = r2_score(y_test, y_pred)
        
        report[name] = score
    
    return report

