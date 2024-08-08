import numpy as np
from typing import Dict, Union
from scipy import stats

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Formula:
    MAE = (1 / N) * Σ |y_i - ŷ_i|

    Where:
    N is the number of samples
    y_i is the i-th actual value
    ŷ_i is the i-th predicted value
    Σ denotes the sum over all samples

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values

    Returns:
        float: MAE value
    """
    return np.mean(np.abs(actual - predicted))

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).

    Formula:
    RMSE = sqrt((1 / N) * Σ (y_i - ŷ_i)^2)

    Where:
    N is the number of samples
    y_i is the i-th actual value
    ŷ_i is the i-th predicted value
    Σ denotes the sum over all samples
    sqrt denotes the square root

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values

    Returns:
        float: RMSE value
    """
    return np.sqrt(np.mean((actual - predicted)**2))

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Formula:
    MAPE = (100% / N) * Σ |y_i - ŷ_i| / |y_i|

    Where:
    N is the number of samples
    y_i is the i-th actual value
    ŷ_i is the i-th predicted value
    Σ denotes the sum over all samples
    | | denotes the absolute value

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values

    Returns:
        float: MAPE value
    """
    return 100 * np.mean(np.abs((actual - predicted) / actual))

def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    Formula:
    sMAPE = (200% / N) * Σ |y_i - ŷ_i| / (|y_i| + |ŷ_i|)

    Where:
    N is the number of samples
    y_i is the i-th actual value
    ŷ_i is the i-th predicted value
    Σ denotes the sum over all samples
    | | denotes the absolute value

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values

    Returns:
        float: sMAPE value
    """
    return 200 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))

def rmae(actual: np.ndarray, predicted: np.ndarray, naive_forecast: np.ndarray) -> float:
    """
    Calculate Relative Mean Absolute Error (rMAE).

    Formula:
    rMAE = MAE(actual, predicted) / MAE(actual, naive_forecast)

    Where:
    MAE(a, b) is the Mean Absolute Error between a and b
    naive_forecast is typically the previous day's actual values

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        naive_forecast (np.ndarray): Naive forecast values

    Returns:
        float: rMAE value
    """
    return mae(actual, predicted) / mae(actual, naive_forecast)

def calculate_all_metrics(actual: np.ndarray, predicted: np.ndarray, naive_forecast: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        naive_forecast (np.ndarray): Naive forecast values

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics
    """
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "rMAE": rmae(actual, predicted, naive_forecast)
    }

