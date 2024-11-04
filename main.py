# main.py

from src.data_processing import load_data, preprocess_data, plot_correlation_matrix
from src.model_training import train_model
from src.model_evaluation import evaluate_model, plot_feature_importance, plot_shap_values

def main():
    # Load the data
    data = load_data('data/employee_data.csv')
    
    # Display correlation matrix for initial data understanding
    plot_correlation_matrix(data)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
    
    print("Model Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    
    # Plot feature importance
    feature_names = X_train.columns
    plot_feature_importance(model, feature_names)
    
    # Plot SHAP values
    plot_shap_values(model, X_train)

if __name__ == "__main__":
    main()
