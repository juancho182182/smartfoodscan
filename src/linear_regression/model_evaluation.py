from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model on the test data using multiple metrics.

    Parameters:
    - model: A trained model object with a predict() method.
    - X_test: Features for the test set.
    - y_test: Actual target values for the test set.

    Returns:
    - A tuple containing MSE, RMSE, MAE, and R² metrics.
    """

    if X_test is None or y_test is None:
        raise ValueError("Test data (X_test, y_test) cannot be None.")

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Test data (X_test, y_test) cannot be empty.")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n")

    return rmse, mse, mae, r2
