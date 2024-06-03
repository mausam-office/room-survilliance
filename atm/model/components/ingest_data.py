import pandas as pd
from model.src.data_ingesting import IngestData


def ingest_data(data_filepath: str) -> pd.DataFrame:
    """
    Ingesting the data from the file.

    Args:
        data_filepath: path to the file.
    Returns: 
        pandas Dataframe
    """
    try:
        in_data = IngestData(data_filepath)
        df = in_data.load_data()
        return df
    except Exception as e:
        print(f"Error in loading data: {e}")
        raise e
