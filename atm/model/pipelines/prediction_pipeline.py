import os
import joblib

import streamlit as st

from utils.save_dataset import CSVDataset
from model.components.clean_data import clean_data

def prediction_pipeline(df,model_name):
    X_input, _ = clean_data(df, False, True)
    model_path = os.path.join('atm/saved_model' , model_name + '.pkl')
    model = joblib.load(open(model_path,'rb'))
    model_prediction = model.predict(X_input)
    if (model_prediction == 0 ):
        result = 'sitted'
    else:
        result = 'standing'

    return result

