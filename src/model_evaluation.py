# src/model_evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, cm, report

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the Random Forest model."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.show()

def plot_shap_values(model, X_train):
    """Plot SHAP values to interpret the model's predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Summary plot of SHAP values
    shap.summary_plot(shap_values[1], X_train, plot_type="bar")
    shap.summary_plot(shap_values[1], X_train)