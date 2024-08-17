import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any

import optuna
import pandas as pd
import pytz
from optuna.storages import RDBStorage

from evaluation_utils import calculate_all_metrics


class EstimatorManager:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.db_path = os.path.join(results_dir, 'tuning', 'optuna_master.db')
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.db_url = f"sqlite:///{self.db_path}"
        self.storage = RDBStorage(self.db_url)

    def get_storage(self):
        return self.storage


class Estimator(ABC):
    def __init__(self, name: str, manager: EstimatorManager):
        self.name = name
        self.manager = manager
        self.hyperparameters = {}
        self.last_optimization_date = None
        self.optimization_frequency = timedelta(days=30)
        self.performance_threshold = 0.1
        self.eval_metric = "MAE"
        self.best_performance = float('inf')
        self.utc = pytz.UTC
        self.fit_time = 0
        self.predict_time = 0
        self.optimize_time = 0

    @abstractmethod
    def fit(self, train_data: Dict[str, pd.DataFrame]):
        pass

    @abstractmethod
    def predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        pass

    def timed_fit(self, train_data: Dict[str, pd.DataFrame]):
        start_time = time.time()
        self.fit(train_data)
        self.fit_time = time.time() - start_time

    def timed_predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        start_time = time.time()
        predictions = self.predict(test_data)
        self.predict_time = time.time() - start_time
        return predictions

    @abstractmethod
    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split the training data into training and validation sets.

        :param train_data: Dictionary containing raw training data
        :return: Dictionary with 'train' and 'valid' keys, each containing a data dictionary
        """
        pass

    def optimize(self, train_data: Dict[str, pd.DataFrame], n_trials: int = 50, current_date: datetime = None):
        storage = self.manager.get_storage()
        study_name = f"{self.name}_optimization_{current_date.strftime('%Y%m%d_%H%M%S')}"

        start_time = time.time()

        split_data = self.split_data(train_data)
        train_subset = split_data['train']
        valid_subset = split_data['valid']

        def objective(trial):
            params = self.define_hyperparameter_space(trial)
            self.hyperparameters = params
            prepared_train_data = self.prepare_data(train_subset)
            self.fit(prepared_train_data)
            prepared_valid_data = self.prepare_data(valid_subset)
            predictions = self.predict(prepared_valid_data)
            metrics = calculate_all_metrics(predictions.values, valid_subset['day_ahead_prices'].values,
                                            valid_subset['naive_forecast'].values)
            return metrics[self.eval_metric]

        study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)
        study.optimize(objective, n_trials=n_trials)

        self.hyperparameters = study.best_params
        self.last_optimization_date = datetime.now(self.utc)
        self.best_performance = study.best_value

        optimization_time = time.time() - start_time

        # Log the essential information
        study.set_user_attr('estimator_name', self.name)
        study.set_user_attr('optimization_date', current_date.isoformat())
        study.set_user_attr('optimization_time', optimization_time)
        study.set_user_attr('best_params', study.best_params)
        study.set_user_attr('best_value', study.best_value)

        # Calculate metrics for the best trial
        best_trial_params = study.best_params
        self.hyperparameters = best_trial_params
        prepared_valid_data = self.prepare_data(valid_subset)
        best_predictions = self.predict(prepared_valid_data)
        best_metrics = calculate_all_metrics(best_predictions.values, valid_subset['day_ahead_prices'].values,
                                             valid_subset['naive_forecast'].values)
        study.set_user_attr('best_metrics', best_metrics)

        self.optimize_time = optimization_time

    def get_execution_times(self):
        return {
            'fit_time': self.fit_time,
            'predict_time': self.predict_time,
            'optimize_time': self.optimize_time,
            'total_time': self.fit_time + self.predict_time + self.optimize_time
        }

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