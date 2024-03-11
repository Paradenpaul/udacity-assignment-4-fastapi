import os
import sys
import inspect
import pytest
import pandas as pd
import numpy as np

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from model import train_model, compute_model_metrics, inference
from data import process_data


# ---- CREATE FIXTURES ---------
@pytest.fixture(scope="module")
def load_data():
    # Path to the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Construct the path to the data directory
    data_dir = os.path.join(dir_path, '..', 'data')

    # Path to the census_cleaned.csv file
    data_path = os.path.join(data_dir, "census_cleaned.csv")

    # Import the data
    data = pd.read_csv(data_path)
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
    assert isinstance(
        preds, np.ndarray), "Inference should return a numpy array."


def test_compute_model_metrics(processed_data):
    X, y = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(
        precision, float) and isinstance(
        recall, float) and isinstance(
            fbeta, float), "Metrics should be float."
