import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import f1_score
from model.src.model_eval import Accuracy, Precision, Recall, F1
from typing import Tuple

def eval_model(
        model: ClassifierMixin, 
        X_test: pd.DataFrame|np.ndarray,
        y_test: pd.Series|np.ndarray
) -> Tuple:
    """
    Evaluates model on test data.
    Args:
        model: RegressorMixin 
        X_test: pd.DataFrame
        y_test: pd.Series|np.ndarray
    Returns:
        float: rmse
        float: mse
        float: r2
    """

    try:
        prediction = model.predict(X_test) # type: ignore
        accuracy = Accuracy().calculate_scores(y_test, prediction)
        precision = Precision().calculate_scores(y_test, prediction)
        recall = Recall().calculate_scores(y_test, prediction)
        f1 = F1().calculate_scores(y_test, prediction)

        return accuracy, precision, recall, f1
    except Exception as e:
        print(f"Error calculating scores.\n {e}")
        raise e