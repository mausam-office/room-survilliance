from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import Tuple
from typing_extensions import Annotated

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class DuplicateDataRemovalStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate data.
        Args:
            data: pd.DataFrame
        """
        if data.duplicated().sum() > 0:
            data.drop_duplicates(inplace=True)
        return data


class FeatureRemovalStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """
        Removes redundant features.
        Args:
            data: pd.DataFrame        
        """
        try:
            cols_with_start_end_angles = self.arc_angle_features(data)
            coorelated_features_to_drop = self.coorelated_features()
            less_important_features_to_drop = self.less_important_features()

            columns_to_drop = [
                'img_w', 'img_h', 'datetime',
                *coorelated_features_to_drop, 
                *less_important_features_to_drop, 
                *cols_with_start_end_angles
            ]
            data.drop(columns_to_drop, axis=1, inplace=True)
            return data
        except Exception as e:
            print(f"Error in FeatureRemoval {e}")
            raise e

    def arc_angle_features(self, data):
        # cols_without_start_end_angles = []
        cols_with_start_end_angles = []
        for col in data.columns:
            if "start_angle" in col or "end_angle" in col:
                cols_with_start_end_angles.append(col)
                continue
            # cols_without_start_end_angles.append(col)
        return cols_with_start_end_angles
    
    def coorelated_features(self):
        return [
            # 'angle_knee_r',             # corelated with angle in waist and required ankle for appropriate calculation
            'angle_knee_l',             # corelated with angle in waist and required ankle for appropriate calculation
            'dist_height_r',            # corelated and required ankle for appropriate calculation
            'dist_height_l',            # corelated and required ankle for appropriate calculation
            # 'height_knee_shoulder_r',   # corelated with height_waist_knee_r/l
            'height_knee_shoulder_l',   # corelated with height_waist_knee_r/l
            # 'height_ankle_waist_r',     # corelated with height_waist_knee_r/l
            # 'height_ankle_waist_l',     # corelated with height_waist_knee_r/l
            # 'dist_width',               # corelated with shoulder_l_r
            # 'waist_l_r',                # corelated with shoulder_l_r
            'visibility_waist_r',       # corelated with visibility_waist_l
            'visibility_wrist_l',       # corelated with visibility_elbow_l
            'visibility_wrist_r',       # corelated with visibility_elbow_r
            'visibility_ankle_l',       # corelated with visibility_knee_l	
            'visibility_ankle_r',       # corelated with visibility_knee_r	
        ]

    def less_important_features(self):
        return [
            'visibility_ear_r',
            'visibility_ear_l',
        ]


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """
        Preprocesses the data. (Here Label encoding.)
        Args:
            df: pandas.Dataframe
        """
        try:
            label_encoder = LabelEncoder()
            data['label'] = label_encoder.fit_transform(data['label'])
            return data
        except Exception as e:
            print(f"Error in preprocessing the data: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame, test_size=0.2) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Splits the data in train and test sets.

        Args:
            df: pandas.Dataframe
        """
        try:
            X = data.drop(['label'], axis=1)
            y = data['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"Error in preprocessing the data: {e}")
            raise e
        
class DataCleaning:
    """
    Prepocessing the data and then splits the data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        """
        Handling the data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            print(f"Error in preprocessing the data: {e}")
            raise e

