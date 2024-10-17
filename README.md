# Loan Risk Model Prediction

This repository contains a complete machine learning pipeline, from data ingestion and preprocessing to model training, evaluation, and deployment using FastAPI. The pipeline also includes unit and integration tests to ensure the robustness of each component.

## Pipeline Diagram
![Pipeline Diagram](Pipeline_Diagram.png)


## Project Structure

```
ml_pipeline_project/
│
├── main.py              # Entry point for the FastAPI app (Deployment)
├── pipeline.py          # Contains the main pipeline logic
├── data_ingestion.py    # Handles data ingestion and validation
├── preprocessing.py     # Handles data preprocessing and feature engineering
├── model_training.py    # Handles model building and training
├── model_evaluation.py  # Handles model evaluation and metrics
├── config.py            # Configuration settings and hyperparameters
├── utils.py             # Utility functions (e.g., logging, data checks)
├── tests/               # Directory for unit and integration tests
│   ├── test_pipeline.py
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── ...
├── requirements.txt     # List of dependencies
├── Dockerfile           # Dockerfile for containerization
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI/CD pipeline configuration

```


## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml_pipeline_project.git
cd ml_pipeline_project
```


### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```
### 3. Install the Dependencies
```bash
pip install -r requirements.txt
```

## Running the Pipeline
To run the machine learning pipeline (including data ingestion, preprocessing, model training, and evaluation), execute the following command:

```bash
python pipeline.py
```
This will preprocess the data, perform hyperparameter tuning, train the model, and save the final model to the models/ directory.

## Running the FastAPI Application (Deployment)
To deploy the FastAPI application for making predictions:

```bash
uvicorn main:app --reload
```
This will start the FastAPI server, which you can access at http://127.0.0.1:8000.


## Available Endpoints
- /predict: POST endpoint to make predictions. Example request body:

```json

{
  "loanAmount": 3000,
  "apr": 199,
  "nPaidOff": 0,
  "isFunded": 1,
  "state": "CA",
  "leadCost": 0,
  "payFrequency": "B",
  "originallyScheduledPaymentAmount": 6395.19
}
```
- /health: GET endpoint for health checks.


## Running the Tests
To run the unit and integration tests:

```bash
pytest
```
