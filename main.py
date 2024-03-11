import os
import joblib
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load your trained model and any necessary preprocessors
# Path to the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model", "trained_model.joblib")
encoder_path = os.path.join(current_dir, "model", "encoder.joblib")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)  # Example for categorical encoding


# ----------------- PYDANTIC TYPE CHECKER -----------------------
class PredictionItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        allow_population_by_field_name = True


# ------------------- WELCOME MESSAGE ----------------------
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Income Prediction API"}


# ------------ SINGLE INSTANCE INFERENCE ---------------------
@app.post("/predict/")
async def make_prediction(item: PredictionItem = Body(...)):
    try:
        # Convert Pydantic model to dict
        input_data = item.dict()

        # Categorical features for encoding
        categorical_features = [
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'native_country']

        # Assuming these are your numerical features based on the earlier model
        # description
        numerical_features = [
            'age',
            'fnlgt',
            'education_num',
            'capital_gain',
            'capital_loss',
            'hours_per_week']

        # Extract and prepare categorical data for encoding
        categorical_data = [input_data[feature]
                            for feature in categorical_features]
        categorical_data_reshaped = np.array([categorical_data]).reshape(1, -1)

        # Encode the categorical data
        encoded_categorical_data = encoder.transform(categorical_data_reshaped)

        # Extract numerical data
        numerical_data = np.array([input_data[feature]
                                  for feature in numerical_features]).reshape(1, -1)

        # Combine encoded categorical data with numerical data for the model
        # input
        model_input = np.concatenate(
            [numerical_data, encoded_categorical_data], axis=1)

        # Make prediction
        prediction = model.predict(model_input)

        # Convert the prediction to a meaningful response
        prediction_label = "Income > 50K" if prediction[0] == 1 else "Income <= 50K"

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
