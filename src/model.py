import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def train_linear_regression(X_train, y_train):
    """
    Trains an Ordinary Least Squares (OLS) Linear Regression model.
    
    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target vector.
        
    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluates the model performance on the test set.
    
    Parameters:
        model (LinearRegression): Trained model.
        X_test (pd.DataFrame): Testing feature matrix.
        y_test (pd.Series): Testing target vector.
        
    Returns:
        dict: Performance metrics (MAE, RMSE, R2, intercept, coefficients).
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'intercept': model.intercept_,
        'coef': model.coef_
    }
    return metrics

def save_model(model, filepath="reports/journey_time_model.joblib"):
    """
    Serializes and saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved successfully to {filepath}")

def load_model(filepath="reports/journey_time_model.joblib"):
    """
    Loads a serialized model from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)
