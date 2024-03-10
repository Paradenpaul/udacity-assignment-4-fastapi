import os
import sys
import inspect
from fastapi.testclient import TestClient

# Set the working directory to the project root so relative paths load correctly
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from main import app

client = TestClient(app)


# -------------- TEST GET METHODS ---------------------------
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Income Prediction API"}  


# -------------- TEST POST METHODS --------------------------
def test_predict_income_greater_than_50k():
    # Example data expected to predict "Income > 50K"
    data = {
        "age": 52,
        "workclass": "Private",
        "fnlgt": 23453,  
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income > 50K"}


def test_predict_income_less_or_equal_50k():
    # Example data expected to predict "Income <= 50K"
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 23453,  
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30,
        "native_country": "United-States"
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Income <= 50K"}


