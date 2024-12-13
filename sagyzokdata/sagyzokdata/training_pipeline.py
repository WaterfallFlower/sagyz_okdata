import pandas as pd
from implmod.pytorch_impl import train_autoencoder, detect_anomalies
from implmod.xgboost_impl import train_xgb
from implmod.tensorflow_impl import build_tf_model, train_tf_model
import numpy as np

data = pd.read_csv('data/processed_data.csv')

X = data.drop('future_spend', axis=1)
y = data['future_spend']

# 1. Тренируем модель PyTorch для аномалий
pytorch_model = train_autoencoder(X, epochs=5)
anomalies = detect_anomalies(pytorch_model, X)

# Фильтруем данные от аномалий для следующего шага, если это нужно
X_clean = X[~anomalies]
y_clean = y[~anomalies]

# 2. Тренируем XGBoost для регрессии (прогнозирование будущих расходов)
xgb_model = train_xgb(X_clean, y_clean)

# 3. Тренируем TensorFlow модель для временных рядов (предполагая, что данные упорядочены по времени)
# Пример: X_tf shape (samples, timesteps, features)
X_tf = np.reshape(X_clean.values, (X_clean.shape[0], 1, X_clean.shape[1]))
y_tf = y_clean.values

tf_model = build_tf_model((1, X_clean.shape[1]))
tf_model = train_tf_model(tf_model, X_tf, y_tf, epochs=3)