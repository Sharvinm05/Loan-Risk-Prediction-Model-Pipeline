import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..pipeline import main_pipeline
from ..preprocessing import build_preprocessing_pipeline, preprocess_data
from ..data_ingestion import ingest_data
from ..model_training import build_model
from ..model_evaluation import evaluate_model


def test_pipeline():
    # Run the entire pipeline
    try:
        main_pipeline()
        assert True, "Pipeline ran successfully"
    except Exception as e:
        assert False, f"Pipeline encountered an error: {e}"

def test_model_training():
    df = ingest_data('data/loan.csv')
    preprocessor = build_preprocessing_pipeline()
    X, y = preprocess_data(df, preprocessor)
    
    model = build_model(X, y, params=None)
    assert model is not None, "Model training failed"
    
def test_model_evaluation():
    df = ingest_data('data/loan.csv')
    preprocessor = build_preprocessing_pipeline()
    X, y = preprocess_data(df, preprocessor)
    
    model = build_model(X, y, params=None)
    metrics = evaluate_model(model, X, y)
    
    assert metrics['AUC'] > 0.5, "AUC score is lower than expected"
