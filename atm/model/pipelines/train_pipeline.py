from model.components.ingest_data import ingest_data
from model.components.clean_data import clean_data
from model.components.train_model import train_model
from model.components.eval_model import eval_model
from model.components.config import ModelNameConfig

def train_pipeline(data_filepath: str):
    """
    Pipeline for training data.
    Args:
        data_filepath: str, path to data file 
    """
    df = ingest_data(data_filepath)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test, ModelNameConfig())
    accuracy, precision, recall, f1 = eval_model(model, X_test, y_test)
    print(f"""
        {accuracy=},
        {precision=},
        {recall=},
        {f1=}
    """)