import tensorflow as tf
from tensorflow.keras import layers, models

def build_tf_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_tf_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model