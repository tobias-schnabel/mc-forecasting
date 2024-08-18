import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import pytz
from tqdm import tqdm

from data_loader import DataLoader
from estimator import Estimator
from evaluation_utils import calculate_all_metrics


class ForecastEngine:
    """
    ForecastEngine is responsible for running forecasts using multiple estimators on the provided data.

    Attributes:
        data_loader (DataLoader): An instance of DataLoader to load and manage data.
        estimators (List[Estimator]): A list of Estimator instances to be used for forecasting.
        results_dir (str): Directory path where results will be saved.
        days_since_optimization (Dict[str, int]): Tracks the number of days since each estimator was last optimized.
        results (Dict[str, pd.DataFrame]): Stores the results of the forecasts for each estimator.
        hyperparameter_history (Dict[str, pd.DataFrame]): Stores the history of hyperparameters for each estimator.
        utc (pytz.UTC): UTC timezone object for handling datetime localization.
        _min_train_window (timedelta): Minimum training window duration.
        _max_train_window (Optional[timedelta]): Maximum training window duration.
    """


    def __init__(self, data_loader: DataLoader, estimators: List[Estimator]):
        """
        Initializes the ForecastEngine with the given data loader, estimators, and results directory.

        Args:
            data_loader (DataLoader): An instance of DataLoader to load and manage data.
            estimators (List[Estimator]): A list of Estimator instances to be used for forecasting.
        """
        # add a check that all estimators have the same results_dir
        for estimator in estimators:
            if estimator.manager.results_dir != estimators[0].manager.results_dir:
                raise ValueError("All estimators must have the same results directory")
        self.data_loader = data_loader
        self.data_loader.load_data()
        self.estimators = estimators
        self.results_dir = self.estimators[0].manager.results_dir  # Use the results directory of the first estimator
        self.estimators = estimators
        self.days_since_optimization = {estimator.name: 0 for estimator in estimators}
        self.results = {estimator.name: pd.DataFrame() for estimator in estimators}
        self.hyperparameter_history = {estimator.name: pd.DataFrame() for estimator in estimators}
        self.utc = pytz.UTC
        self._min_train_window = timedelta(days=1)  # Default minimum of 1 day
        self._max_train_window = None

    @property
    def min_train_window(self) -> timedelta:
        """
        Gets the minimum training window.

        Returns:
            timedelta: The minimum training window as a timedelta object.
        """
        return self._min_train_window

    @min_train_window.setter
    def min_train_window(self, value: int):
        """
        Sets the minimum training window.

        Args:
            value (int): An integer representing the number of days for the minimum training window.

        Raises:
            ValueError: If the value is not an integer, is less than 1, or if it is greater than the maximum training window.
        """
        if not isinstance(value, int):
            raise ValueError("min_train_window must be an integer representing days")
        if value < 1:
            raise ValueError("min_train_window must be a at least 1")
        if self._max_train_window and timedelta(days=value) > self._max_train_window:
            raise ValueError("min_train_window cannot be greater than max_train_window")
        self._min_train_window = timedelta(days=value)

    @property
    def max_train_window(self) -> Optional[timedelta]:
        """
        Gets the maximum training window.

        Returns:
            Optional[timedelta]: The maximum training window as a timedelta object, or None if not set.
        """
        return self._max_train_window

    @max_train_window.setter
    def max_train_window(self, value: Optional[int]):
        """
        Sets the maximum training window.

        Args:
            value (Optional[int]): A positive integer representing the number of days for the maximum training window, or None to unset.

        Raises:
            ValueError: If the value is not a positive integer or if it is less than the minimum training window.
        """
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("max_train_window must be a positive integer representing days")
            value = timedelta(days=value)
        if value and self._min_train_window > value:
            raise ValueError("max_train_window cannot be less than min_train_window")
        self._max_train_window = value

    def run_forecast(self, start_date: datetime, end_date: datetime):
        """
        Runs the forecast for the given date range using the provided estimators.

        Args:
            start_date (datetime): The start date for the forecast.
            end_date (datetime): The end date for the forecast.
        """
        # noinspection PyArgumentList
        start_date_utc = self.utc.localize(dt=start_date) if start_date.tzinfo is None else start_date
        # noinspection PyArgumentList
        end_date_utc = self.utc.localize(dt=end_date) if end_date.tzinfo is None else end_date
        current_date = start_date_utc + self.min_train_window

        total_steps = (end_date_utc - start_date_utc).days * len(self.estimators)
        with tqdm(total=total_steps, desc="Forecasting Progress") as pbar:
            if self.min_train_window > timedelta(days=1):
                pbar.update(int(self.min_train_window.days))  # Skip the first day if min_train_window > 1
            while current_date < end_date_utc:
                train_start = self.data_loader.data_min_date
                if self.max_train_window:
                    train_start = max(train_start, current_date - self.max_train_window)
                train_data = self.data_loader.get_slice(train_start, current_date)
                test_data = self.data_loader.get_next_day_with_naive(current_date - timedelta(days=1))

                for estimator in self.estimators:
                    recent_performance = self._get_recent_performance(estimator)
                    if estimator.should_optimize(current_date, recent_performance):
                        self._optimize_estimator(estimator, train_data, current_date)
                        self.days_since_optimization[estimator.name] = 0
                    else:
                        self.days_since_optimization[estimator.name] += 1

                    # Ensure hyperparameters are applied before fitting
                    if estimator.hyperparameters:
                        estimator.set_model_params(**estimator.hyperparameters)
                    prepared_train_data = estimator.prepare_data(train_data)
                    prepared_test_data = estimator.prepare_data(test_data)
                    estimator.timed_fit(prepared_train_data)
                    predictions = estimator.timed_predict(prepared_test_data)

                    self.save_results(estimator, current_date, predictions, test_data)
                    self.save_hyperparameters(estimator, current_date)

                    self.days_since_optimization[estimator.name] += 1
                    pbar.update(1)

                current_date += timedelta(days=1)

            pbar.update(1)  # Update progress bar to 100%
            self._save_final_results()
            self._save_final_hyperparameters()
            for estimator in self.estimators:
                estimator.last_optimization_date = None  # Reset last optimization date

    def _optimize_estimator(self, estimator: Estimator, train_data: Dict[str, pd.DataFrame], current_date: datetime):
        """
        Optimizes the given estimator if necessary based on recent performance.

        Args:
            estimator (Estimator): The estimator to be optimized.
            train_data (Dict[str, pd.DataFrame]): The training data.
            current_date (datetime): The current date in the forecast process.
        """
        recent_performance = self._get_recent_performance(estimator)
        if estimator.should_optimize(current_date, recent_performance):
            # noinspection PyTypeChecker
            estimator.optimize(train_data, current_date)

    def _get_recent_performance(self, estimator: Estimator) -> float:
        """
        Retrieves the recent performance of the given estimator.

        Args:
            estimator (Estimator): The estimator whose performance is to be retrieved.

        Returns:
            float: The mean performance metric of the estimator over the last 3 days.
        """
        recent_results = self.results[estimator.name].tail(3)  # Consider last 3 days
        if len(recent_results) > 0:
            return recent_results[f'{estimator.eval_metric}'].mean()
        return float('inf')

    def save_results(self, estimator: Estimator, forecast_date: datetime, predictions: pd.DataFrame,
                     test_data: Dict[str, pd.DataFrame]):
        """
        Saves the results of the forecast for the given estimator and date.

        Args:
            estimator (Estimator): The estimator used for the forecast.
            forecast_date (datetime): The date of the forecast.
            predictions (pd.DataFrame): The predictions made by the estimator.
            test_data (Dict[str, pd.DataFrame]): The actual test data.
        """
        actuals = test_data['day_ahead_prices']
        naive_forecast = test_data['naive_forecast']

        metrics = calculate_all_metrics(predictions.values, actuals.values, naive_forecast.values)

        # Update the best performance if current performance is better
        current_performance = metrics[estimator.eval_metric]
        if current_performance < estimator.best_performance:
            estimator.best_performance = current_performance

        execution_times = estimator.get_execution_times()

        result_row = pd.DataFrame({
            'forecast_date': [forecast_date],
            **{f'{k}': [v] for k, v in metrics.items()},
            **{f'{k}': [v] for k, v in execution_times.items()}
        })

        self.results[estimator.name] = pd.concat([self.results[estimator.name], result_row], ignore_index=True)


    def save_hyperparameters(self, estimator: Estimator, forecast_date: datetime):
        """
        Saves the hyperparameters of the given estimator for the forecast date.
        """
        if estimator.hyperparameters:
            hyper_row = pd.DataFrame({
                'forecast_date': [forecast_date],
                **{f'param_{k}': [v] for k, v in estimator.hyperparameters.items() if estimator.hyperparameters}
            })
            self.hyperparameter_history[estimator.name] = pd.concat(
                [self.hyperparameter_history[estimator.name], hyper_row],
                ignore_index=True
            )


    def _save_final_results(self):
        """
        Saves the final results of the forecasts to the results directory.
        """
        for estimator_name, results_df in self.results.items():
            file_path = os.path.join(self.results_dir, f"{estimator_name}_results.parquet")
            results_df.to_parquet(file_path, index=False)
            print(f"Results for {estimator_name} saved to {estimator_name}_results.parquet")


    def _save_final_hyperparameters(self):
        """
        Saves the final hyperparameters of the forecasts to the results directory.
        """
        for estimator_name, hyper_df in self.hyperparameter_history.items():
            if not hyper_df.empty:
                file_path = os.path.join(self.results_dir, 'tuning', f"{estimator_name}_hyperparameters.parquet")
                hyper_df.to_parquet(file_path, index=False)
                print(f"Hyperparameters for {estimator_name} saved to {estimator_name}_hyperparameters.parquet")

