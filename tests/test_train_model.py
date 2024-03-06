import os
import sys
import inspect
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
 
from data import process_data
from model import train_model, compute_model_metrics, inference


# ---- CREATE FIXTURES ---------
@pytest.fixture(scope="module")
def load_data():
    
    # gives the path of demo.py 
    path = os.path.realpath(__file__) 
   
    # gives the directory where demo.py  
    # exists 
    dir = os.path.dirname(path) 
  
    # replaces folder name of Sibling_1 to  
    # Sibling_2 in directory 
    dir = dir.replace('tests', 'data') 
  
    # changes the current directory to  
    #  Sibling_2 folder 
    os.chdir(dir)  
  
    # import the data fixture 
    data = pd.read_csv(os.path.join(dir, "census_cleaned.csv"))
    return data


@pytest.fixture(scope="module")
def processed_data(load_data):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    label = 'salary'

    X, y, _, _ = process_data(
        load_data, 
        categorical_features=cat_features, 
        label=label, 
        training=True  # Assuming we want to process the whole dataset for training
    )
    return X, y

# ----- RUN THE FUNCTION TESTS ------
def test_train_model(processed_data):
    X, y = processed_data
    model = train_model(X, y)
    assert hasattr(model, 'predict'), "Model doesn't have the predict method."


def test_inference(processed_data):
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray), "Inference should return a numpy array."


def test_compute_model_metrics(processed_data):
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float) and isinstance(recall, float) and isinstance(fbeta, float), "Metrics should be float."
