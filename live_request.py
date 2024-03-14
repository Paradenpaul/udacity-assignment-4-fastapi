import requests


base_url = 'https://udacity-assignment-4-fastapi.onrender.com'
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
predict_url = f'{base_url}/predict/'
response = requests.post(predict_url, json=data)
print("POST /predict/ response:", response.json())

# GET request to the root endpoint
root_url = base_url + '/'
root_response = requests.get(root_url)
print("GET / status code:", root_response.status_code)