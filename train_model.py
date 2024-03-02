import pandas as pd
import numpy as np
from data import process_data  # Import the process_data function
from model import train_model, compute_model_metrics, inference  # Assuming model.py is also in the same directory
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
df = pd.read_csv('data/census_cleaned.csv')

# Define your features and label
categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
        ]

label = "salary"

# Split the dataset into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=categorical_features, 
    label=label, 
    training=True
)

# Process the testing data
X_test, y_test, _, _ = process_data(
    test, 
    categorical_features=categorical_features, 
    label=label, 
    training=False, 
    encoder=encoder, 
    lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Compute model metrics
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}, Recall: {recall}, F-beta: {fbeta}")

# Save the model and encoders to disk for later use
joblib.dump(model, 'model/trained_model.joblib')
joblib.dump(encoder, 'model/encoder.joblib')
joblib.dump(lb, 'model/label_binarizer.joblib')
