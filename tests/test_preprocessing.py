import pytest
from ..preprocessing import build_preprocessing_pipeline, preprocess_data
from ..data_ingestion import ingest_data
import numpy as np
import os
import joblib

def test_preprocessing():
    # Ingest data
    df = ingest_data('data/loan.csv')

    # Count rows before preprocessing
    original_row_count = df.shape[0]
    
    # Count how many rows have missing 'loanStatus'
    missing_loan_status_count = df['loanStatus'].isna().sum()

    # Preprocess the data
    df_preprocessed = preprocess_data(df)

    # Check if preprocessing output is as expected
    assert df_preprocessed is not None, "Preprocessed data returned None"
    
    # Calculate expected row count after removing rare classes
    class_counts = df['loanStatus'].value_counts()
    rows_to_remove = class_counts[class_counts <= 1].sum()
    expected_row_count = original_row_count - missing_loan_status_count - rows_to_remove

    # Check the row count after preprocessing
    assert df_preprocessed.shape[0] == expected_row_count, (
        f"Number of rows in preprocessed data does not match expected count. "
        f"Expected {expected_row_count}, but got {df_preprocessed.shape[0]}."
    )

    # Check if LabelEncoders are saved correctly
    assert os.path.exists('models/label_encoder.pkl'), "Label encoder for 'loanStatus' was not saved"
    assert os.path.exists('models/state_encoder.pkl'), "Label encoder for 'state' was not saved"

    # Load and verify the encoders
    loan_status_encoder = joblib.load('models/label_encoder.pkl')
    state_encoder = joblib.load('models/state_encoder.pkl')

    # Ensure the encoders are not None
    assert loan_status_encoder is not None, "Failed to load 'loanStatus' LabelEncoder"
    assert state_encoder is not None, "Failed to load 'state' LabelEncoder"

    # Check if the LabelEncoder for 'loanStatus' works correctly
    assert isinstance(loan_status_encoder.classes_, np.ndarray), "'loanStatus' LabelEncoder does not contain classes"
    assert len(loan_status_encoder.classes_) > 0, "'loanStatus' LabelEncoder contains no classes"

    # Check if the LabelEncoder for 'state' works correctly
    assert isinstance(state_encoder.classes_, np.ndarray), "'state' LabelEncoder does not contain classes"
    assert len(state_encoder.classes_) > 0, "'state' LabelEncoder contains no classes"

    # Verify that 'loanStatus' and 'state' columns have been transformed into integers
    assert df_preprocessed['loanStatus'].dtype in [np.int32, np.int64], "'loanStatus' column was not encoded correctly"
    assert df_preprocessed['state'].dtype in [np.int32, np.int64], "'state' column was not encoded correctly"

    # Check if feature engineering was applied correctly
    assert 'is_monthly_payment' in df_preprocessed.columns, "'is_monthly_payment' feature was not created"
    assert 'loan_to_payment_ratio' in df_preprocessed.columns, "'loan_to_payment_ratio' feature was not created"

    print("All preprocessing tests passed successfully.")

