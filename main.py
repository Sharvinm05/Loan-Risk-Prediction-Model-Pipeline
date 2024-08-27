from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from config import MODEL_PATH, LABEL_ENCODER_PATH, STATE_ENCODER_PATH, PREPROCESSOR_PATH
from pipeline import map_loan_status_to_risk
import logging
import uvicorn

app = FastAPI()

# Load the trained model, encoders, and preprocessing pipeline
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
state_encoder = joblib.load(STATE_ENCODER_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Configure logging for monitoring
logging.basicConfig(filename='model_logs.log', level=logging.INFO)

def preprocess_input(input_data: dict):
    df = pd.DataFrame([input_data])
    
    # Apply the full preprocessing pipeline to match the training process
    processed_data = preprocessor.transform(df)
    
    return processed_data

@app.post('/predict')
async def predict(loan_application: dict):
    try:
        # Preprocess the input data
        data = preprocess_input(loan_application)
        
        # Make prediction
        prediction_proba = model.predict(data)
        prediction_numeric = int(prediction_proba.argmax())
        prediction_status = label_encoder.inverse_transform([prediction_numeric])[0]
        
        # Map to risk category
        prediction_risk = map_loan_status_to_risk(prediction_status)
        
        # Log and return the result
        logging.info(f'Input: {loan_application}')
        logging.info(f'Prediction made: Status: {prediction_status}, Risk: {prediction_risk}')
        return {"loanStatus": prediction_status, "riskCategory": prediction_risk}
    
    except Exception as e:
        logging.error(f'Error in prediction: {e}', exc_info=True)
        raise HTTPException(status_code=400, detail="Prediction failed")

@app.get('/health')
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
