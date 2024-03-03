import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compute_model_metrics, inference

# ---- CREATE FIXTURES ---------
@pytest.fixture(scope="module")
def load_data():
    # Adjust the path to where your census_cleaned.csv is located
    path_to_csv = 'census_cleaned.csv'
    data = pd.read_csv(path_to_csv)
    return data

@pytest.fixture(scope="module")
def processed_data(load_data):
    categorical_features = ['your', 'categorical', 'feature', 'names']  # Update this list
    label = 'your_label_column_name'  # Update this with your actual label column name
    X, y, _, _ = process_data(
        load_data, 
        categorical_features=categorical_features, 
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