from src.preprocessing import get_preprocessed_data
from src.train_models import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model


if __name__ == "__main__":

    file_path = "data/diabetes_health_indicators.csv"

    X_train, X_test, y_train, y_test, scaler = get_preprocessed_data(file_path)

    print("\n===== Logistic Regression =====")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)

    print("\n===== Random Forest =====")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)