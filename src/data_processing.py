# src/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load employee attrition data from a CSV file."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """Preprocess the data by handling categorical variables and splitting into train/test sets."""
    # Assume 'Attrition' is the target and other columns are features
    X = data.drop(columns='Attrition')
    y = data['Attrition']
    
    # Convert categorical variables into dummy/indicator variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

def plot_correlation_matrix(data):
    """Plot a correlation matrix to understand relationships between features."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Convert categorical variables into dummy/indicator variables
    data_numeric = pd.get_dummies(data, drop_first=True)
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = data_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Features")
    plt.show()
