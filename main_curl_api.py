import requests

url = 'http://localhost:8000/predict/'
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 2345,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post(url, json=data)
print(response.json())
