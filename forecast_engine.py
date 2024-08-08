import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from evaluation_utils import calculate_all_metrics


class DataLoader:
    """
    Handles loading and slicing of data from Parquet files in a specified folder.
    """

    def __init__(self, data_folder: str):
        """
        Initialize the DataLoader with the path to the data folder.
        :param data_folder: Path to the folder containing Parquet files
        """
        self.data_folder = data_folder
        self.data = {}
        self.installed_capacity = None
        self.installed_capacity_year = None

    def load_data(self):
        """
        Load all Parquet files from the specified folder.
        """
        installed_capacity_files = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.parquet'):
                key = os.path.splitext(file)[0]
                file_path = os.path.join(self.data_folder, file)

                if key.startswith('installed_capacity_'):
                    installed_capacity_files.append((key, file_path))
                else:
                    self.data[key] = pd.read_parquet(file_path)
                    # Ensure datetime index is timezone-aware (UTC)
                    if not self.data[key].index.tz:
                        self.data[key].index = self.data[key].index.tz_localize('UTC')

        # Handle installed capacity data
        if installed_capacity_files:
            # Sort files by year (descending) and use the most recent
            installed_capacity_files.sort(key=lambda x: int(re.findall(r'\d+', x[0])[0]), reverse=True)
            key, file_path = installed_capacity_files[0]
            self.installed_capacity = pd.read_parquet(file_path)
            self.installed_capacity_year = int(re.findall(r'\d+', key)[0])
            print(f"Using installed capacity data from {self.installed_capacity_year}")

    def get_slice(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get a slice of data for all loaded datasets within the specified date range.
        :param start_date: Start of the date range
        :param end_date: End of the date range
        :return: Dictionary of sliced DataFrames
        """
        sliced_data = {key: df[(df.index >= start_date) & (df.index < end_date)]
                       for key, df in self.data.items()}
        # Add installed capacity data
        if self.installed_capacity is not None:
            sliced_data['installed_capacity'] = self.installed_capacity
        return sliced_data

    def get_next_day(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get all data for the next day (midnight to 23:00) after the specified date.
        :param date: The reference date
        :return: Dictionary of DataFrames for the next day
        """
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = start_date + timedelta(days=1)
        return self.get_slice(start_date, end_date)

    def get_next_day_features(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get feature data for the next day (midnight to 23:00) after the specified date, excluding day-ahead prices.
        :param date: The reference date
        :return: Dictionary of DataFrames for the next day, excluding day-ahead prices
        """
        next_day_data = self.get_next_day(date)
        next_day_data.pop('day_ahead_prices', None)
        return next_day_data


class Estimator(ABC):
    """
    Abstract base class for all estimators (forecasting models).
    """

    def __init__(self, name: str):
        """
        Initialize the Estimator.
        :param name: Name of the estimator
        """
        self.name = name
        self.hyperparameters = {}
        self.parameter_history = pd.DataFrame()

    @abstractmethod
    def fit(self, train_data: Dict[str, pd.DataFrame]):
        """
        Fit the model on training data.
        :param train_data: Dictionary of training DataFrames
        """
        pass

    @abstractmethod
    def predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Make predictions on test data.
        :param test_data: Dictionary of test DataFrames
        :return: DataFrame of predictions
        """
        pass

    def update_hyperparameters(self, new_hyperparameters: Dict[str, Any], date: datetime):
        """
        Update the hyperparameters of the model and record the change.
        :param new_hyperparameters: Dictionary of new hyperparameter values
        :param date: Date of the hyperparameter update
        """
        self.hyperparameters.update(new_hyperparameters)
        record = pd.DataFrame([{**{'date': date}, **new_hyperparameters}])
        self.parameter_history = pd.concat([self.parameter_history, record])

    @abstractmethod
    def optimize(self, recent_performance: pd.DataFrame):
        """
        Perform walk-forward optimization based on recent performance.
        :param recent_performance: DataFrame of recent performance metrics
        """
        pass


class ForecastEngine:
    """
    Manages the entire forecasting process, including running estimators and evaluating results.
    """

    def __init__(self, data_loader: DataLoader, estimators: List[Estimator]):
        """
        Initialize the ForecastEngine.
        :param data_loader: DataLoader instance
        :param estimators: List of Estimator instances
        """
        self.data_loader = data_loader
        self.estimators = estimators
        self.results = pd.DataFrame()
        self.evaluation_results = pd.DataFrame()

    def run_forecast(self, start_date: datetime, end_date: datetime, train_window: Optional[timedelta] = None):
        """
        Run the forecasting process for all estimators over the specified time range.
        :param start_date: Start date for forecasting
        :param end_date: End date for forecasting
        :param train_window: Optional time window for training data. If None, use all available data.
        """
        current_date = start_date
        with tqdm(total=(end_date - start_date).days, desc="Forecasting Progress") as pbar:
            while current_date < end_date:
                if train_window:
                    train_start = current_date - train_window
                else:
                    train_start = self.data_loader.data['day_ahead_prices'].index.min()
                train_end = current_date
                train_data = self.data_loader.get_slice(train_start, train_end)
                test_features = self.data_loader.get_next_day_features(current_date)
                test_prices = self.data_loader.get_next_day(current_date)['day_ahead_prices']

                for estimator in self.estimators:
                    forecast, execution_time = self._run_estimator(estimator, train_data, test_features)
                    self._save_results(estimator.name, current_date, forecast, execution_time,
                                       estimator.hyperparameters)
                    naive_forecast = self.data_loader.get_slice(current_date - timedelta(days=1), current_date)[
                        'day_ahead_prices']
                    self._evaluate_forecast(estimator.name, current_date, forecast, test_prices, naive_forecast)

                    # Perform walk-forward optimization
                    recent_performance = self.evaluation_results[
                        (self.evaluation_results['estimator'] == estimator.name) &
                        (self.evaluation_results['forecast_date'] == current_date)
                        ]
                    estimator.optimize(recent_performance)

                current_date += timedelta(days=1)
                pbar.update(1)

    def _run_estimator(self, estimator: Estimator, train_data: Dict[str, pd.DataFrame],
                       test_data: Dict[str, pd.DataFrame]) -> (pd.DataFrame, float):
        """
        Run a single estimator on the given train and test data.
        :param estimator: Estimator instance
        :param train_data: Training data
        :param test_data: Test data
        :return: Tuple of (forecast DataFrame, execution time)
        """
        start_time = datetime.now()
        estimator.fit(train_data)  # This method can be parallelized or use GPU if implemented in the specific estimator
        forecast = estimator.predict(test_data)
        execution_time = (datetime.now() - start_time).total_seconds()
        return forecast, execution_time

    def _save_results(self, estimator_name: str, forecast_date: datetime,
                      forecast: pd.DataFrame, execution_time: float, hyperparameters: Dict[str, Any]):
        """
        Save the results of a single forecast run.
        :param estimator_name: Name of the estimator
        :param forecast_date: Date of the Forecast
        :param forecast: Forecast DataFrame
        :param execution_time: Execution time of the forecast
        :param hyperparameters: Current hyperparameters of the estimator
        """
        result = forecast.copy()
        result['estimator'] = estimator_name
        result['forecast_date'] = forecast_date
        result['execution_time'] = execution_time
        for key, value in hyperparameters.items():
            result[f'param_{key}'] = value
        self.results = pd.concat([self.results, result])

    def _evaluate_forecast(self, estimator_name: str, forecast_date: datetime,
                           forecast: pd.DataFrame, actual: pd.DataFrame, naive_forecast: pd.DataFrame):
        """
        Evaluate the forecast against actual values using multiple metrics.

        :param estimator_name: Name of the estimator
        :param forecast_date: Date of the forecast
        :param forecast: Forecast DataFrame
        :param actual: Actual values DataFrame
        :param naive_forecast: Naive forecast DataFrame (typically previous day's values)
        """
        # Ensure all inputs are numpy arrays
        forecast_array = forecast.values.flatten()
        actual_array = actual.values.flatten()
        naive_forecast_array = naive_forecast.values.flatten()

        # Calculate all metrics
        metrics = calculate_all_metrics(actual_array, forecast_array, naive_forecast_array)

        # Create a DataFrame with the results
        eval_result = pd.DataFrame({
            'estimator': [estimator_name],
            'forecast_date': [forecast_date],
            'MAE': [metrics['MAE']],
            'RMSE': [metrics['RMSE']],
            'MAPE': [metrics['MAPE']],
            'sMAPE': [metrics['sMAPE']],
            'rMAE': [metrics['rMAE']]
        })

        # Concatenate with existing results
        self.evaluation_results = pd.concat([self.evaluation_results, eval_result], ignore_index=True)


class ResultsManager:
    """
    Handles saving and loading of forecast and evaluation results.
    """

    def __init__(self, results: pd.DataFrame, evaluation_results: pd.DataFrame):
        """
        Initialize the ResultsManager.
        :param results: DataFrame of forecast results
        :param evaluation_results: DataFrame of evaluation results
        """
        self.results = results
        self.evaluation_results = evaluation_results

    def save_results(self, results_path: str, evaluation_path: str):
        """
        Save forecast and evaluation results to Parquet files.
        :param results_path: Path to save forecast results
        :param evaluation_path: Path to save evaluation results
        """
        self.results.to_parquet(results_path)
        self.evaluation_results.to_parquet(evaluation_path)

    def load_results(self, results_path: str, evaluation_path: str):
        """
        Load forecast and evaluation results from Parquet files.
        :param results_path: Path to load forecast results from
        :param evaluation_path: Path to load evaluation results from
        """
        self.results = pd.read_parquet(results_path)
        self.evaluation_results = pd.read_parquet(evaluation_path)
