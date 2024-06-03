from model.components.ingest_data import ingest_data
from model.components.clean_data import clean_data
from model.components.train_model import train_model
from model.components.eval_model import eval_model
from model.components.config import ModelConfig

def train_pipeline(data_filepath: list[str]):
    """
    Pipeline for training data.
    Args:
        data_filepath: str, path to data file 
    """
    if ModelConfig.seperate_test_set:
        assert len(data_filepath) == 2, "When using seperate test set two seperate data paths needs to be provided."
        
        df_train = ingest_data(data_filepath[0])
        df_test = ingest_data(data_filepath[1])

        X_train, y_train = clean_data(df_train) # type: ignore
        X_test, y_test = clean_data(df_test) # type: ignore
    else:
        df = ingest_data(data_filepath[0])
        X_train, X_test, y_train, y_test = clean_data(df) # type: ignore

    model = train_model(X_train, y_train, ModelConfig(), early_stopping=ModelConfig.early_stopping, random_state=ModelConfig.random_state) # type: ignore
    accuracy, precision, recall, f1 = eval_model(model, X_test, y_test) # type: ignore
    print(f"""
        {accuracy=},
        {precision=},
        {recall=},
        {f1=}
    """)