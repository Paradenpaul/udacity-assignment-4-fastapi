import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, compute_model_metrics, inference



# ---------------- SLICE EVALUATION -------------
def evaluate_model_on_slices(df, feature_to_slice, model, encoder, lb, categorical_features):
    """Evaluates the model on slices of data based on unique values of a specified feature."""
    unique_values = df[feature_to_slice].unique()
    results = []
    
    for value in unique_values:
        df_slice = df[df[feature_to_slice] == value]
        # Ensure all categorical features are passed here, not just the one being sliced
        X_slice, y_slice, _, _ = process_data(
            df_slice, 
            categorical_features=categorical_features,  # Pass the full list of categorical features
            label='salary',  # Assuming 'salary' is the label column
            training=False, 
            encoder=encoder, 
            lb=lb
        )
        preds_slice = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        results.append((value, precision, recall, fbeta))
    
    return results


def MLapp(): 

    # Load your dataset
    df = pd.read_csv('data/census_cleaned.csv')

    # Define your features and label
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

    # Split the dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label=label, 
        training=True
    )

    # Process the testing data
    X_test, y_test, _, _ = process_data(
        test, 
        categorical_features=cat_features, 
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

    # Assuming `df` is your DataFrame, `model` is your trained model,
    # `encoder` and `lb` are fitted OneHotEncoder and LabelBinarizer from your training phase
    feature_to_slice = 'workclass'  # The feature you're evaluating

    slice_results = evaluate_model_on_slices(df, feature_to_slice, model, encoder, lb, cat_features)


    # Output the results to a file
    with open('slice_validation/slice_output_{}.txt'.format(feature_to_slice), 'w') as f:
        for value, precision, recall, fbeta in slice_results:
            f.write(f"{feature_to_slice}={value}: Precision={precision}, Recall={recall}, F1={fbeta}\n")


# --------------------------------------------------------------
if __name__ == "__main__":
    MLapp()
