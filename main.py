from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from config import MODEL_PATH, LABEL_ENCODER_PATH, STATE_ENCODER_PATH, PREPROCESSOR_PATH
from pipeline import map_loan_status_to_risk
from security_models import LoanApplicationInput, PredictionResponse, mask_sensitive_data, generate_request_id
from security_middleware import (
    SecurityMiddleware, RateLimitMiddleware, ContentLengthMiddleware,
    verify_api_key, setup_security_logging, validate_model_file_integrity
)
import logging
import uvicorn

app = FastAPI(
    title="Loan Risk Prediction API",
    description="Secure API for loan risk prediction with input validation and authentication",
    version="1.0.0"
)

# Add security middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=3600)
app.add_middleware(ContentLengthMiddleware, max_size=10240)  # 10KB limit

# Configure CORS with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost", "https://127.0.0.1"],  # Only specific origins
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Setup secure logging
security_logger = setup_security_logging()

# Configure application logging (avoid logging to files in production without proper rotation)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model loading with security validation
def load_models_securely():
    """Securely load ML models with integrity checks"""
    models = {}
    model_paths = {
        'model': MODEL_PATH,
        'label_encoder': LABEL_ENCODER_PATH,
        'state_encoder': STATE_ENCODER_PATH,
        'preprocessor': PREPROCESSOR_PATH
    }
    
    for name, path in model_paths.items():
        if not validate_model_file_integrity(path):
            raise RuntimeError(f"Model file integrity check failed for {name}: {path}")
        
        try:
            models[name] = joblib.load(path)
            logger.info(f"Successfully loaded {name} from {path}")
        except Exception as e:
            logger.error(f"Failed to load {name} from {path}: {e}")
            raise RuntimeError(f"Failed to load {name}: {e}")
    
    return models

# Load models securely at startup
try:
    models = load_models_securely()
    model = models['model']
    label_encoder = models['label_encoder']
    state_encoder = models['state_encoder']
    preprocessor = models['preprocessor']
    logger.info("All models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load models: {e}")
    raise RuntimeError("Application cannot start due to model loading failure")

def preprocess_input(input_data: LoanApplicationInput) -> pd.DataFrame:
    """Securely preprocess validated input data"""
    try:
        # Convert Pydantic model to dictionary
        data_dict = input_data.dict()
        df = pd.DataFrame([data_dict])
        
        # Apply the full preprocessing pipeline to match the training process
        processed_data = preprocessor.transform(df)
        
        return processed_data
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)[:100]}...")  # Limit error message length
        raise ValueError("Data preprocessing failed")

@app.post('/predict', response_model=PredictionResponse)
async def predict(
    loan_application: LoanApplicationInput,
    user: str = Depends(verify_api_key)
):
    """
    Predict loan risk with secure input validation and authentication
    
    Requires X-API-Key header for authentication
    """
    request_id = generate_request_id(loan_application.dict())
    
    try:
        # Preprocess the validated input data
        data = preprocess_input(loan_application)
        
        # Make prediction
        prediction_proba = model.predict(data)
        prediction_numeric = int(prediction_proba.argmax())
        prediction_status = label_encoder.inverse_transform([prediction_numeric])[0]
        
        # Map to risk category
        prediction_risk = map_loan_status_to_risk(prediction_status)
        
        # Calculate confidence score (max probability)
        confidence = float(prediction_proba.max()) if hasattr(prediction_proba, 'max') else None
        
        # Secure logging with masked data
        masked_input = mask_sensitive_data(loan_application.dict())
        security_logger.info(
            f"Prediction request - ID: {request_id}, User: {user}, "
            f"Masked Input: {masked_input}"
        )
        logger.info(
            f"Prediction completed - ID: {request_id}, Status: {prediction_status}, "
            f"Risk: {prediction_risk}, Confidence: {confidence}"
        )
        
        return PredictionResponse(
            loanStatus=prediction_status,
            riskCategory=prediction_risk,
            confidence=confidence
        )
    
    except ValueError as ve:
        # Handle validation errors
        security_logger.warning(f"Validation error - ID: {request_id}, Error: {str(ve)[:100]}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input data"
        )
    except Exception as e:
        # Handle unexpected errors without exposing system details
        security_logger.error(f"Prediction error - ID: {request_id}, Error: {str(e)[:100]}")
        logger.error(f'Prediction failed for request {request_id}: {str(e)[:100]}...')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service temporarily unavailable"
        )

@app.get('/health')
async def health_check():
    """Health check endpoint (no authentication required)"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": pd.Timestamp.now().isoformat(),
            "models_loaded": all([
                model is not None,
                label_encoder is not None,
                state_encoder is not None,
                preprocessor is not None
            ])
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)[:100]}")
        return {"status": "unhealthy", "timestamp": pd.Timestamp.now().isoformat()}

if __name__ == "__main__":
    # Secure configuration for production
    host = os.getenv("HOST", "127.0.0.1")  # Default to localhost only
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}, reload={reload}")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        reload=reload,
        access_log=False  # Disable access logs to prevent log flooding
    )
