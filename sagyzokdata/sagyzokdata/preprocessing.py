import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    # Загрузка транзакций
    transactions = pd.read_csv('data/transactions.csv')
    bloomberg = pd.read_csv('data/bloomberg_data.csv')
    factors = pd.read_csv('data/additional_factors.csv')
    
    data = transactions.merge(bloomberg, on='date', how='left')
    data = data.merge(factors, on='date', how='left')
    
    # Очистка данных
    data.dropna(inplace=True)

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

if __name__ == "__main__":
    processed_data = load_data()
    processed_data.to_csv('data/processed_data.csv', index=False)