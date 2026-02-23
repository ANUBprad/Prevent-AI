from src.preprocessing import get_preprocessed_data

if __name__ == "__main__":
    file_path = "data/diabetes_health_indicators.csv"

    X_train, X_test, y_train, y_test, scaler = get_preprocessed_data(file_path)

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    print("\nTrain Class Distribution:")
    print(y_train.value_counts())

    print("\nPreprocessing Successful ✅")