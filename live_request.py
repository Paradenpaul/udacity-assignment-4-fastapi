import requests


url = 'https://udacity-assignment-4-fastapi.onrender.com/predict/'
data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 23453,
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

# POST request to the /predict/ endpoint
response = requests.post(url, json=data)

# Print both the JSON response and the status code of the POST request
print("Model inference result:", response.json())
print("Status code of /predict/:", response.status_code)
