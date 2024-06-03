import os
import pandas as pd

class IngestData:
    def __init__(self, data_filepath: str) -> None:
        """
        Args:
            data_filepath: string, path to data file 

        Returns:
            None
        """
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"{data_filepath} doesn't exists.")
        self.data_filepath = data_filepath

    def load_data(self):
        print(f"Loading data from csv file at {self.data_filepath}.")
        return pd.read_csv(self.data_filepath)
    