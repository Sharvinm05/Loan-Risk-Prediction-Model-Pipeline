import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import DEFAULT_MODEL_PARAMS
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

def stratified_split_with_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in split.split(X_resampled, y_resampled):
        X_train, X_val = X_resampled[train_index], X_resampled[val_index]
        y_train, y_val = y_resampled[train_index], y_resampled[val_index]
    
    return X_train, X_val, y_train, y_val


def objective(trial, X_train, y_train):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # Increased range
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),  # Wider range
        'max_depth': trial.suggest_int('max_depth', 3, 10),  # Added max_depth
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),  # Added feature_fraction
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),  # Added bagging_fraction
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),  # Added bagging_freq
        'num_class': len(np.unique(y_train)),
        'n_jobs': -1,
        'class_weight': 'balanced',  # Handle class imbalance
        'device': 'gpu'  # Use GPU for faster training
    }
    
    # Initialize stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loglosses = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Apply SMOTE to the training data to handle class imbalance
        smote = SMOTE(random_state=42)
        X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_tr_resampled, label=y_tr_resampled)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # Train the model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
        )
        
        # Predict on the validation set
        y_pred = model.predict(X_val)
        
        # Calculate log loss for this fold
        logloss = log_loss(y_val, y_pred)
        loglosses.append(logloss)
    
    # Return the mean log loss across all folds
    return np.mean(loglosses)


def perform_hyperparameter_tuning(X_train, y_train, n_trials=60):  # Increased from 50 to 100
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    best_params = study.best_params
    best_params['objective'] = 'multiclass'
    best_params['metric'] = 'multi_logloss'
    best_params['num_class'] = len(np.unique(y_train))
    best_params['n_jobs'] = -1
    
    return best_params
