import lightgbm as lgb
import joblib
from config import MODEL_PATH

def build_model(X, y, params):
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=200)
    
    joblib.dump(model, MODEL_PATH)
    return model
