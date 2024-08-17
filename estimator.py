import os
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import optuna
import pytz
from abc import ABC, abstractmethod
from evaluation_utils import calculate_all_metrics


class Estimator(ABC):
    def __init__(self, name: str, base_dir: str):
        self.name = name
        self.base_dir = base_dir
        self.hyperparameters = {}
        self.last_optimization_date = None
        self.optimization_frequency = timedelta(days=30)
        self.performance_threshold = 0.1
        self.eval_metric = "MAE"
        self.best_performance = float('inf')
        self.utc = pytz.UTC

    @abstractmethod
    def fit(self, train_data: Dict[str, pd.DataFrame]):
        """
        Fit the model on the training data.

        :param train_data: Dictionary containing training data
        """
        pass

    @abstractmethod
    def predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Make predictions using the fitted model.

        :param test_data: Dictionary containing test data
        :return: DataFrame with predictions
        """
        pass

    @abstractmethod
    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Prepare the data for fitting or prediction.

        :param data: Dictionary containing raw data
        :return: Dictionary containing prepared data
        """
        pass

    @abstractmethod
    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter space for optimization.

        :param trial: Optuna trial object
        :return: Dictionary of hyperparameters
        """
        pass

    def calculate_all_performance_metrics(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                                          naive_forecast: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate all performance metrics.

        :param predictions: DataFrame of predictions
        :param actuals: DataFrame of actual values
        :param naive_forecast: DataFrame of naive forecast
        :return: Dictionary of metric names and values
        """
        return calculate_all_metrics(predictions, actuals, naive_forecast)

    def optimize(self, train_data: Dict[str, pd.DataFrame], valid_data: Dict[str, pd.DataFrame], n_trials: int = 50):
        def objective(trial):
            params = self.define_hyperparameter_space(trial)
            self.hyperparameters = params
            prepared_train_data = self.prepare_data(train_data)
            self.fit(prepared_train_data)
            prepared_valid_data = self.prepare_data(valid_data)
            predictions = self.predict(prepared_valid_data)
            metrics = self.calculate_all_performance_metrics(predictions, valid_data['day_ahead_prices'],
                                                             valid_data['naive_forecast'])
            return metrics[self.eval_metric]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.hyperparameters = study.best_params
        self.last_optimization_date = datetime.now(self.utc)
        self.best_performance = study.best_value

    def should_optimize(self, current_date: datetime, recent_performance: float) -> bool:
        if self.last_optimization_date is None:
            return True
        time_condition = (current_date - self.last_optimization_date) >= self.optimization_frequency
        performance_condition = recent_performance > (1 + self.performance_threshold) * self.best_performance
        return time_condition or performance_condition

    def set_optimization_params(self, frequency: timedelta, threshold: float, metric: str):
        self.optimization_frequency = frequency
        self.performance_threshold = threshold
        self.eval_metric = metric