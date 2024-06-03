from model.components.ingest_data import ingest_data
from model.components.clean_data import clean_data
from model.components.train_model import train_model
from model.components.eval_model import eval_model
from model.components.config import ModelConfig

import joblib

def train_pipeline(data_filepath: list[str], model_name='', random_state=0, use_ui=False, seperate_test_set=False, output_model_name=''):
    """
    Pipeline for training data.
    Args:
        data_filepath: str, path to data file 
    """
    
    if not use_ui:
        seperate_test_set = ModelConfig.seperate_test_set
        early_stopping = ModelConfig.early_stopping
        random_state = ModelConfig.random_state
        model_name = ModelConfig.model_name
    else:
        early_stopping = True
    
    if seperate_test_set:
        assert len(data_filepath) == 2, "When using seperate test set two seperate data paths needs to be provided."
        
        df_train = ingest_data(data_filepath[0])
        df_test = ingest_data(data_filepath[1])

        X_train, y_train = clean_data(df_train, True) # type: ignore
        X_test, y_test = clean_data(df_test, True) # type: ignore
    else:
        df = ingest_data(data_filepath[0])
        X_train, X_test, y_train, y_test = clean_data(df, seperate_test_set) # type: ignore
    
    model = train_model(X_train, y_train, model_name, early_stopping=early_stopping, random_state=random_state) # type: ignore
    accuracy, precision, recall, f1 = eval_model(model, X_test, y_test) # type: ignore
    print(f"""
        {accuracy=},
        {precision=},
        {recall=},
        {f1=}
    """)

    joblib.dump(model, f'./atm/saved_model/{output_model_name}.pkl')
    print('Model saved')
