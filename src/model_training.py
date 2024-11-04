# src/model_training.py

from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """Train a Random Forest model on the training data."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
