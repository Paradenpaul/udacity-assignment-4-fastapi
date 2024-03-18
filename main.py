import os
import joblib
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

# Load the trained model and any necessary preprocessors at startup
model = None
encoder = None

@app.on_event("startup")
async def startup_event():
    global model, encoder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "trained_model.joblib")
    encoder_path = os.path.join(current_dir, "model", "encoder.joblib")
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

# Function to convert hyphens to underscores
def hyphen_to_underscore(field_name: str) -> str:
    return field_name.replace("-", "_")

# Pydantic model with alias generator
class PredictionItem(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=23455)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Income Prediction API"}

@app.post("/predict/")
async def make_prediction(item: PredictionItem = Body(...)):
    try:
        # Convert Pydantic model to dict with aliases
        input_data = item.dict(by_alias=True)

        # Categorical features for encoding
        categorical_features = [
            'workclass', 'education', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'native_country']

        # Numerical features
        numerical_features = [
            'age', 'fnlgt', 'education_num',
            'capital_gain', 'capital_loss', 'hours_per_week']

        # Extract and prepare categorical data for encoding
        categorical_data = [input_data[feature] for feature in categorical_features]
        categorical_data_reshaped = np.array([categorical_data]).reshape(1, -1)

        # Encode the categorical data
        encoded_categorical_data = encoder.transform(categorical_data_reshaped)

        # Extract numerical data
        numerical_data = np.array([input_data[feature] for feature in numerical_features]).reshape(1, -1)

        # Combine encoded categorical data with numerical data for the model input
        model_input = np.concatenate([numerical_data, encoded_categorical_data], axis=1)

        # Make prediction
        prediction = model.predict(model_input)

        # Convert the prediction to a meaningful response
        prediction_label = "Income > 50K" if prediction[0] == 1 else "Income <= 50K"

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
