"""
Simple linear regression example using scikit-learn.

Trains on the built-in diabetes dataset (BMI feature) to predict disease
progression. Structured for serving on Domino Data Lab: `train_model()`
fits the model, and `predict()` is the inference entry point Domino's
Model API can call.
"""

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train_model(test_size=0.2, random_state=42):
    """
    Train a linear regression model on the diabetes dataset (BMI feature).

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        The fitted model.
    metrics : dict
        Mean squared error and R^2 score on the held-out test set.
    """
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:, [2]]  # BMI column
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    return model, metrics


# Train once at import time so the model is ready when Domino loads this file.
model, training_metrics = train_model()


def predict(data):
    """
    Inference entry point for Domino's Model API.

    Parameters
    ----------
    data : float, list of float, or array-like of shape (n_samples, 1)
        New BMI value(s) to score.

    Returns
    -------
    list of float
        Predicted disease progression value(s).
    """
    X_new = np.array(data, dtype=float).reshape(-1, 1)
    return model.predict(X_new).tolist()


if __name__ == "__main__":
    print(f"Coefficient (slope): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Mean squared error: {training_metrics['mse']:.2f}")
    print(f"R^2 score: {training_metrics['r2']:.2f}")

    example_input = [0.05, -0.02, 0.1]
    print(f"Example predictions for {example_input}: {predict(example_input)}")