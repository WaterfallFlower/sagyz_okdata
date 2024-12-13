import xgboost as xgb
import numpy as np

def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    return model