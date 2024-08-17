import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytz
from data_loader import DataLoader
from estimator import Estimator
from evaluation_utils import calculate_all_metrics
from estimator import EstimatorManager


class ForecastEngine:
    def __init__(self, data_loader: DataLoader, estimators: List[Estimator], results_dir: str):
        self.data_loader = data_loader
        self.data_loader.load_data()
        self.estimators = estimators
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.estimator_manager = EstimatorManager(results_dir)  # Create EstimatorManager
        # Pass the manager to each estimator
        for estimator in estimators:
            estimator.manager = self.estimator_manager
        self.estimators = estimators
        self.results = {estimator.name: pd.DataFrame() for estimator in estimators}
        self.utc = pytz.UTC

    def run_forecast(self, start_date: datetime, end_date: datetime, train_window: Optional[timedelta] = None,
                     optimize_frequency: int = 30):
        start_date_utc = self.utc.localize(start_date) if start_date.tzinfo is None else start_date
        end_date_utc = self.utc.localize(end_date) if end_date.tzinfo is None else end_date
        current_date = start_date_utc + timedelta(days=1)

        days_since_optimization = {estimator.name: optimize_frequency for estimator in self.estimators}

        total_steps = (end_date_utc - start_date_utc).days * len(self.estimators)
        with tqdm(total=total_steps, desc="Forecasting Progress") as pbar:
            while current_date < end_date_utc:
                if train_window:
                    train_start = max(self.data_loader.data_min_date, current_date - train_window)
                else:
                    train_start = self.data_loader.data_min_date
                train_data = self.data_loader.get_slice(train_start, current_date)
                test_data = self.data_loader.get_next_day_with_naive(current_date)

                for estimator in self.estimators:
                    if days_since_optimization[
                        estimator.name] >= optimize_frequency:  # Check if it's time to optimize the estimator
                        self._optimize_estimator(estimator, train_data, test_data)
                        days_since_optimization[estimator.name] = 0

                    estimator.fit(train_data)
                    predictions = estimator.predict(test_data)

                    self.save_results(estimator, current_date, predictions, test_data)

                    days_since_optimization[estimator.name] += 1
                    pbar.update(1)

                current_date += timedelta(days=1)
            pbar.update(1)  # Update progress bar to 100%
            self._save_final_results()

    def _optimize_estimator(self, estimator: Estimator, train_data: Dict[str, pd.DataFrame],
                            valid_data: Dict[str, pd.DataFrame]):
        recent_performance = self._get_recent_performance(estimator)
        if estimator.should_optimize(datetime.now(self.utc), recent_performance):
            estimator.optimize(train_data, valid_data)

    def _get_recent_performance(self, estimator: Estimator) -> float:
        recent_results = self.results[estimator.name].tail(3)  # Consider last 3 days
        if len(recent_results) > 0:
            return recent_results[f'{estimator.eval_metric}'].mean()
        return float('inf')

    def save_results(self, estimator: Estimator, forecast_date: datetime, predictions: pd.DataFrame,
                     test_data: Dict[str, pd.DataFrame]):
        actuals = test_data['day_ahead_prices']
        naive_forecast = test_data['naive_forecast']

        metrics = calculate_all_metrics(predictions.values, actuals.values, naive_forecast.values)
        execution_times = estimator.get_execution_times()

        result_row = pd.DataFrame({
            'forecast_date': [forecast_date],
            **{f'{k}': [v] for k, v in metrics.items()},
            **{f'{k}': [v] for k, v in execution_times.items()}
        })

        self.results[estimator.name] = pd.concat([self.results[estimator.name], result_row], ignore_index=True)

    def _save_final_results(self):
        for estimator_name, results_df in self.results.items():
            file_path = os.path.join(self.results_dir, f"{estimator_name}_results.parquet")
            results_df.to_parquet(file_path, index=False)
            print(f"Results for {estimator_name} saved to {file_path}")