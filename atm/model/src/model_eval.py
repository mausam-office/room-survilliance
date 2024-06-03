from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        """
        Calculate the scores for the model
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass


class Accuracy(Evaluation):
    """
    Evaluation Strategyg for accuracy.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            print("Calculating accuracy.")
            accuracy = accuracy_score(y_true, y_pred)
            return accuracy
        except Exception as e:
            print(f"Error calculating Accuracy. {e}")
            raise e


class Recall(Evaluation):
    """
    Evaluation Strategyg for recall.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            print("Calculating Recall.")
            recall = recall_score(y_true, y_pred)
            return recall
        except Exception as e:
            print(f"Error calculating Recall. {e}")
            raise e


class Precision(Evaluation):
    """
    Evaluation Strategyg for precision.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            print("Calculating Precision.")
            precision = precision_score(y_true, y_pred)
            return precision
        except Exception as e:
            print(f"Error calculating Precision. {e}")
            raise e


class F1(Evaluation):
    """
    Evaluation Strategyg for f1 score.
    """
    def calculate_scores(self, y_true: pd.Series|np.ndarray, y_pred: np.ndarray):
        try:
            print("Calculating F1.")
            f1 = f1_score(y_true, y_pred)
            return f1
        except Exception as e:
            print(f"Error calculating F1. {e}")
            raise e
        
