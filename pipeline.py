import logging
from data_ingestion import ingest_data
from preprocessing import build_preprocessing_pipeline, preprocess_data
from model_training import build_model
from model_evaluation import evaluate_model
from hyperparameter_tuning import perform_hyperparameter_tuning
from config import DATA_PATH
from utils import setup_logging
import joblib
import pandas as pd


def map_loan_status_to_risk(status):
    high_risk = [
        'Charged Off', 
        'Charged Off Paid Off', 
        'External Collection', 
        'Settled Bankruptcy', 
        'Settlement Paid Off', 
        'Settlement Pending Paid Off', 
        'Rejected'
    ]
    
    medium_risk = [
        'Internal Collection', 
        'Pending Rescind', 
        'Returned Item'
    ]
    
    low_risk = [
        'New Loan', 
        'Paid Off Loan', 
        'Pending Paid Off', 
        'Voided New Loan', 
        'Pending Application', 
        'Pending Application Fee'
    ]
    
    very_low_risk = [
        'CSR Voided New Loan', 
        'Credit Return Void', 
        'Customer Voided New Loan', 
        'Withdrawn Application'
    ]
    
    if status in high_risk:
        return 'High Risk'
    elif status in medium_risk:
        return 'Medium Risk'
    elif status in low_risk:
        return 'Low Risk'
    elif status in very_low_risk:
        return 'Very Low Risk'
    elif pd.isna(status):
        return 'Unknown Risk'
    else:
        return 'Unknown Risk'



def main_pipeline():
    setup_logging()
    df = ingest_data(DATA_PATH)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Build preprocessing pipeline and apply it
    preprocessor = build_preprocessing_pipeline()
    X = preprocessor.fit_transform(df)
    y = df['loanStatus']
    
    # Perform hyperparameter tuning
    best_params = perform_hyperparameter_tuning(X, y)
    logging.info(f'Best Parameters: {best_params}')
    
    # Build and train the model using the best hyperparameters
    model = build_model(X, y, best_params)
    
    # Evaluate the model
    metrics = evaluate_model(model, X, y)
    
    # Log the final metrics
    logging.info(f'Pipeline completed with metrics: {metrics}')
    
    # Save the model and preprocessor
    joblib.dump(model, 'models/final_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')


if __name__ == "__main__":
    main_pipeline()
