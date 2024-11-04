import pandas as pd
from src.data_preprocessing import preprocess_data
from src.train_model import train_and_evaluate

# Load data
data = pd.read_csv('data/employee_data.csv')

# Preprocess data
X, y, _, _ = preprocess_data(data)

# Train and evaluate models
train_and_evaluate(X, y)