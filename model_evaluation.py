import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss
import logging

def evaluate_model(model, X_test, y_test):
    # Get the predicted probabilities
    y_pred_proba = model.predict(X_test)
    
    # Convert probabilities to predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handling edge case where only one class is predicted
    if len(np.unique(y_pred)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
    else:
        auc = None
        logging.warning("AUC could not be calculated. Model predicted only one class.")
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)
    
    # Generate a classification report
    class_report = classification_report(y_test, y_pred)
    
    # Log the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "log_loss": logloss,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }
    
    # Log the classification report and confusion matrix
    logging.info("Classification Report:\n" + class_report)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))
    
    return metrics
