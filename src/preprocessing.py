import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def check_missing(df):
    """
    Check for missing values in dataset.
    """
    missing = df.isnull().sum()
    return missing


def split_features_target(df):
    """
    Separate features and target variable.
    """
    X = df.drop("Diabetes_012", axis=1)
    y = df["Diabetes_012"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split for multi-class classification.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def get_preprocessed_data(file_path):
    """
    Complete preprocessing pipeline.
    Returns scaled train-test data and scaler.
    """
    df = load_data(file_path)

    print("Dataset Shape:", df.shape)
    print("\nClass Distribution:\n", df["Diabetes_012"].value_counts())

    missing = check_missing(df)
    print("\nMissing Values:\n", missing[missing > 0])

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler