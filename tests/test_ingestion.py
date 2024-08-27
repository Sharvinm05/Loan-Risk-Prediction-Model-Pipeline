import pytest
from ..data_ingestion import ingest_data


def test_data_ingestion():
    df = ingest_data('data/loan.csv')
    
    assert df is not None, "Data ingestion returned None"
    assert df.shape[0] > 0, "Dataframe is empty"
    assert 'loanAmount' in df.columns, "Expected column 'loanAmount' not found"
    assert df.isnull().sum().sum() == 0, "Data contains missing values"
