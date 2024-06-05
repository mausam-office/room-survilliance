import pandas as pd

from typing import Tuple
from typing_extensions import Annotated

from model.components.config import ModelConfig
from model.src.data_cleaning import (
    DataCleaning, 
    DuplicateDataRemovalStrategy, 
    FeatureRemovalStrategy, 
    DataPreProcessStrategy, 
    TargetFeatureSplitStrategy, 
    DataSplitStrategy
)


def clean_data(data: pd.DataFrame, seperate_test_set, prediction=False):
    """
    Component for cleaning data.
    Args:
        data: pandas DataFrame
    """
    try:
        dc = DataCleaning(data, DuplicateDataRemovalStrategy())
        data = dc.handle_data()

        dc = DataCleaning(data, FeatureRemovalStrategy())
        data = dc.handle_data()

        dc = DataCleaning(data, DataPreProcessStrategy())
        data = dc.handle_data()

        dc = DataCleaning(data, TargetFeatureSplitStrategy())
        if seperate_test_set or prediction:
            X, y = dc.handle_data()
            return X, y

        dc = DataCleaning(data, DataSplitStrategy())
        X_train, X_test, y_train, y_test = dc.handle_data()

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error in clean_data func {e}")
        raise e