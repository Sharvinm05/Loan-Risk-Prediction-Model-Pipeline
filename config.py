# config.py

import os

DATA_PATH = 'data/loan.csv'
MODEL_PATH = 'models/final_model.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
STATE_ENCODER_PATH = 'models/state_encoder.pkl'

# Default parameters (to be used if not optimizing)
DEFAULT_MODEL_PARAMS = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 30,
    'n_jobs': -1,
    'num_class': None  # set dynamically
}
