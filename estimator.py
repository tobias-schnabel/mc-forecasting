from sacred import Experiment
from sacred.observers import FileStorageObserver
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
        self.ex = Experiment(name)
        self.ex.observers.append(FileStorageObserver(os.path.join(base_dir, name)))
        self.utc = pytz.UTC

        # Add configuration
        @self.ex.config
        def config():
            optimization = False
            n_trials = 50

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

    # @abstractmethod
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
        @self.ex.main
        def optimization_run(_run, n_trials):
            def objective(trial):
                params = self.define_hyperparameter_space(trial)
                self.hyperparameters = params
                prepared_train_data = self.prepare_data(train_data)
                self.fit(prepared_train_data)
                prepared_valid_data = self.prepare_data(valid_data)
                predictions = self.predict(prepared_valid_data)
                metrics = self.calculate_all_performance_metrics(predictions, valid_data['day_ahead_prices'],
                                                                 valid_data['naive_forecast'])
                _run.log_scalar("trial", trial.number)
                for metric_name, metric_value in metrics.items():
                    _run.log_scalar(metric_name, metric_value)
                return metrics[self.eval_metric]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)

            self.hyperparameters = study.best_params
            _run.log_scalar("best_value", study.best_value)
            for param_name, param_value in study.best_params.items():
                _run.log_scalar(f"best_param_{param_name}", param_value)

            self.last_optimization_date = datetime.now(self.utc)
            self.best_performance = study.best_value

        self.ex.run(config_updates={'optimization': True, 'n_trials': n_trials})

    def log_training(self, train_data: Dict[str, pd.DataFrame]):
        @self.ex.main
        def training_run(_run):
            for param_name, param_value in self.hyperparameters.items():
                _run.log_scalar(f"param_{param_name}", param_value)
            _run.log_scalar("train_start_date", train_data['day_ahead_prices'].index.min().timestamp())
            _run.log_scalar("train_end_date", train_data['day_ahead_prices'].index.max().timestamp())

        # self.ex.run(config_updates={'training': True})

    def log_prediction(self, predictions: pd.DataFrame, actuals: pd.DataFrame, naive_forecast: pd.DataFrame):
        @self.ex.main
        def prediction_run(_run):
            metrics = self.calculate_all_performance_metrics(predictions, actuals, naive_forecast)
            for metric_name, metric_value in metrics.items():
                _run.log_scalar(metric_name, metric_value)

        # self.ex.run(config_updates={'prediction': True})

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