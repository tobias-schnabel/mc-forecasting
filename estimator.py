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
    """
    Manages the storage and setup for estimator tuning results.

    Attributes:
        results_dir (str): Directory where results are stored.
        db_path (str): Path to the SQLite database for Optuna storage.
        db_url (str): URL for the SQLite database.
        storage (RDBStorage): Optuna storage object for managing study results. If None, no database is used.
    """

    def __init__(self, results_dir: str, use_db = True):
        """
        Initializes the EstimatorManager with the given results directory.

        Args:
            results_dir (str): Directory where results are stored.
            use_db (bool): Whether to use a sqlite database for Optuna storage.
        """
        self.results_dir = results_dir
        self.use_db = use_db
        if self.use_db:
            self.db_path = os.path.join(results_dir, 'tuning', 'optuna_master.db')
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.db_url = f"sqlite:///{self.db_path}"
            self.storage = RDBStorage(self.db_url)

    def get_storage(self):
        """
        Returns the Optuna storage object.

        Returns:
            RDBStorage: Optuna storage object for managing study results.
        """
        return self.storage if self.use_db else None


class Estimator(ABC):
    """
    Abstract base class for estimators.

    Attributes:
        name (str): The name of the estimator.
        manager (EstimatorManager): The manager responsible for handling estimator-related operations.
        hyperparameters (dict): Dictionary to store hyperparameters.
        last_optimization_date (datetime): The date when the estimator was last optimized.
        optimization_frequency (timedelta): Frequency at which the estimator should be optimized.
        performance_threshold (float): Performance threshold for triggering optimization.
        eval_metric (str): Evaluation metric used for optimization.
        best_performance (float): Best performance achieved by the estimator.
        utc (pytz.UTC): UTC timezone object for handling datetime localization.
        fit_time (float): Time taken to fit the estimator.
        predict_time (float): Time taken to make predictions.
        optimize_time (float): Time taken to optimize the estimator.
    """

    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        """
        Initializes the Estimator with the given name and manager.

        Args:
            name (str): The name of the estimator.
            results_dir (str): Directory where results are stored.
            use_db (bool): Whether to use a sqlite database for Optuna storage.
        """
        self.name = name
        self.manager = EstimatorManager(results_dir, use_db)
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
        """
        Abstract method to fit the estimator to the training data.

        Args:
            train_data (Dict[str, pd.DataFrame]): The training data.
        """
        pass

    @abstractmethod
    def predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Abstract method to make predictions on the test data.

        Args:
            test_data (Dict[str, pd.DataFrame]): The test data.

        Returns:
            pd.DataFrame: The predictions.
        """
        pass

    def timed_fit(self, train_data: Dict[str, pd.DataFrame]):
        """
        Fits the estimator to the training data and records the time taken.

        Args:
            train_data (Dict[str, pd.DataFrame]): The training data.
        """
        start_time = time.time()
        self.fit(train_data)
        self.fit_time = time.time() - start_time

    def timed_predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Makes predictions on the test data and records the time taken.

        Args:
            test_data (Dict[str, pd.DataFrame]): The test data.

        Returns:
            pd.DataFrame: The predictions.
        """
        start_time = time.time()
        predictions = self.predict(test_data)
        self.predict_time = time.time() - start_time
        return predictions

    @abstractmethod
    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split the training data into training and validation sets.

        Args:
            train_data (Dict[str, pd.DataFrame]): Dictionary containing raw training data.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Dictionary with 'train' and 'valid' keys, each containing a data dictionary.
        """
        pass

    def optimize(self, train_data: Dict[str, pd.DataFrame], n_trials: int = 50, current_date: datetime = None):
        """
        Optimizes the estimator using the given training data and number of trials.

        Args:
            train_data (Dict[str, pd.DataFrame]): The training data.
            n_trials (int, optional): Number of optimization trials. Defaults to 50.
            current_date (datetime, optional): The current date. Defaults to None.
        """
        storage = self.manager.get_storage() if self.use_db else None
        study_name = f"{self.name}_optimization_{current_date.strftime('%Y%m%d_%H%M%S')}"

        start_time = time.time()

        split_data = self.split_data(train_data)
        train_subset = split_data['train']
        valid_subset = split_data['valid']
        prepared_train_data = self.prepare_data(train_subset)
        prepared_valid_data = self.prepare_data(valid_subset)

        def objective(trial):
            params = self.define_hyperparameter_space(trial)
            self.hyperparameters = params
            self.fit(prepared_train_data)
            predictions = self.predict(prepared_valid_data)
            metrics = calculate_all_metrics(predictions.values, valid_subset['day_ahead_prices'].values,
                                            valid_subset['naive_forecast'].values)
            return metrics[self.eval_metric]

        if self.use_db:
            study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)
        else:
            study = optuna.create_study(direction="minimize", study_name=study_name)

        study.optimize(objective, n_trials=n_trials)

        self.hyperparameters = study.best_params
        self.last_optimization_date = datetime.now(self.utc)
        self.best_performance = study.best_value

        optimization_time = time.time() - start_time

        # Calculate metrics for the best trial
        best_trial_params = study.best_params
        self.hyperparameters = best_trial_params
        best_predictions = self.predict(prepared_valid_data)
        best_metrics = calculate_all_metrics(best_predictions.values, valid_subset['day_ahead_prices'].values,
                                             valid_subset['naive_forecast'].values)

        # Log
        if self.use_db:
            study.set_user_attr('estimator_name', self.name)
            study.set_user_attr('study_creation_time', datetime.now().isoformat())
            study.set_user_attr('optimization_date', current_date.isoformat())
            study.set_user_attr('optimization_time', optimization_time)
            study.set_user_attr('best_params', study.best_params)
            study.set_user_attr('best_value', study.best_value)
            study.set_user_attr('best_metrics', best_metrics)


        self.optimize_time = optimization_time

    def get_execution_times(self):
        """
        Returns the execution times for fitting, predicting, and optimizing.

        Returns:
            dict: Dictionary containing the execution times.
        """
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

        Args:
            data (Dict[str, pd.DataFrame]): Dictionary containing raw data.

        Returns:
            Dict[str, Any]: Dictionary containing prepared data.
        """
        pass

    @abstractmethod
    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter space for optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Dict[str, Any]: Dictionary of hyperparameters.
        """
        pass

    def should_optimize(self, current_date: datetime, recent_performance: float) -> bool:
        """
        Determines whether the estimator should be optimized based on the current date and recent performance.

        Args:
            current_date (datetime): The current date.
            recent_performance (float): The recent performance of the estimator.

        Returns:
            bool: True if the estimator should be optimized, False otherwise.
        """
        if self.last_optimization_date is None:
            return True
        time_since_last_optimization = current_date - self.last_optimization_date
        if time_since_last_optimization < timedelta(days=1):  # Minimum 1 day between optimizations
            return False
        time_condition = time_since_last_optimization >= self.optimization_frequency
        performance_condition = recent_performance > (1 + self.performance_threshold) * self.best_performance
        return time_condition or performance_condition

    def set_optimization_params(self, frequency: timedelta, threshold: float, metric: str):
        """
        Sets the optimization parameters for the estimator.

        Args:
            frequency (timedelta): Frequency at which the estimator should be optimized.
            threshold (float): Performance threshold for triggering optimization.
            metric (str): Evaluation metric used for optimization.
        """
        self.optimization_frequency = frequency
        self.performance_threshold = threshold
        self.eval_metric = metric
