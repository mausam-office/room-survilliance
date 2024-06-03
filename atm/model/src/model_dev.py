import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression, SGDClassifier

class Model(ABC):
    """Abstract class for all models."""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.
        Args: 
            X_train: Training data features.
            y_train: Training data labels.
        Returns: 
            None
        """
        pass


class LogisticRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model.
        Args: 
            X_train: Training data features.
            y_train: Training data labels.
        Returns: 
            None
        """
        try:
            classification_model = LogisticRegression(**kwargs)
            classification_model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return classification_model
        except Exception as e:
            logging.error(f"Error in training model with error {e}")
            raise e


class SGDClassificationModel(Model):
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model.
        Args: 
            X_train: Training data features.
            y_train: Training data labels.
        Returns: 
            None
        """
        try:
            classification_model = SGDClassifier(**kwargs)
            classification_model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return classification_model
        except Exception as e:
            logging.error(f"Error in training model with error {e}")
            raise e