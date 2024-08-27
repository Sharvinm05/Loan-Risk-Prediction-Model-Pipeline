import pandas as pd
import logging

def ingest_data(input_file):
    df = pd.read_csv(input_file, parse_dates=['applicationDate', 'originatedDate'])
    
    # Drop rows with missing data
    df.dropna(inplace=True)
    
    # Data Validation
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected in the dataset")
    
    required_columns = ['loanAmount', 'apr', 'state']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} is missing from the dataset")
    
    logging.info(f'Data Ingestion completed: {df.shape[0]} rows, {df.shape[1]} columns')
    
    return df
