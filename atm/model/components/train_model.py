import pandas as pd

from model.src.model_dev import LogisticRegressionModel, SGDClassificationModel
from sklearn.base import ClassifierMixin
from model.components.config import ModelConfig


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame|pd.Series,
    model_name: str,
    **kwargs
) -> ClassifierMixin:
    """
    Trains the model.

    Args: 
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.DataFrame
        y_test: pd.DataFrame
        config: ModelNameConfig
    """
    try:
        model = None
        if model_name == 'LinearRegression':
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train, **kwargs)
            return trained_model
        elif model_name == 'SGDClassification':
            model = SGDClassificationModel()
            trained_model = model.train(X_train, y_train, **kwargs)
            return trained_model
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented.")
    except Exception as e:
        print(f"Error in training model. {e}")
        raise e
    