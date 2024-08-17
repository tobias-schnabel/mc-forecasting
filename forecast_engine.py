from sacred import Experiment
from sacred.observers import FileStorageObserver
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import pytz

from data_loader import DataLoader
from estimator import Estimator

class ForecastEngine:
    def __init__(self, data_loader: DataLoader, estimators: List[Estimator], results_dir: str):
        self.data_loader = data_loader
        self.data_loader.load_data()
        self.estimators = estimators
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = {estimator.name: pd.DataFrame() for estimator in estimators}
        self.ex = Experiment('ForecastEngine')
        self.ex.observers.append(FileStorageObserver(os.path.join(results_dir, 'sacred')))
        self.utc = pytz.UTC

        # Add configuration
        @self.ex.config
        def config():
            optimization = False
            start_date = None
            end_date = None
            train_window = None
            optimize_frequency = 30

    def run_forecast(self, start_date: datetime, end_date: datetime, train_window: Optional[timedelta] = None,
                     optimize_frequency: int = 30):
        @self.ex.main
        def forecast_run(_run):
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
                        if days_since_optimization[estimator.name] >= optimize_frequency:
                            self._optimize_estimator(estimator, train_data, test_data)
                            days_since_optimization[estimator.name] = 0

                        estimator.fit(train_data)
                        predictions = estimator.predict(test_data)

                        self._log_and_save_results(estimator, current_date, predictions, test_data)

                        days_since_optimization[estimator.name] += 1
                        pbar.update(1)

                    current_date += timedelta(days=1)

                self.ex.run(config_updates={
                    'start_date': start_date,
                    'end_date': end_date,
                    'train_window': train_window,
                    'optimize_frequency': optimize_frequency
                })

            self._save_final_results()

        self.ex.run()

    def _optimize_estimator(self, estimator: Estimator, train_data: Dict[str, pd.DataFrame],
                            valid_data: Dict[str, pd.DataFrame]):
        recent_performance = self._get_recent_performance(estimator)
        if estimator.should_optimize(datetime.now(self.utc), recent_performance):
            estimator.optimize(train_data, valid_data)

    def _get_recent_performance(self, estimator: Estimator) -> float:
        recent_results = self.results[estimator.name].tail(3)  # Consider last 3 days
        if len(recent_results) > 0:
            return recent_results[f'metric_{estimator.eval_metric}'].mean()
        return float('inf')

    def _log_and_save_results(self, estimator: Estimator, forecast_date: datetime, predictions: pd.DataFrame,
                              test_data: Dict[str, pd.DataFrame]):
        actuals = test_data['day_ahead_prices']
        naive_forecast = test_data['naive_forecast']

        estimator.log_prediction(predictions, actuals, naive_forecast)

        metrics = estimator.calculate_all_performance_metrics(predictions, actuals, naive_forecast)

        result_row = pd.DataFrame({
            'forecast_date': [forecast_date],
            'predictions': [predictions.to_dict()],
            **{f'metric_{k}': [v] for k, v in metrics.items()}
        })

        self.results[estimator.name] = pd.concat([self.results[estimator.name], result_row], ignore_index=True)

    def _save_final_results(self):
        for estimator_name, results_df in self.results.items():
            file_path = os.path.join(self.results_dir, f"{estimator_name}_results.parquet")
            results_df.to_parquet(file_path, index=False)
            print(f"Results for {estimator_name} saved to {file_path}")