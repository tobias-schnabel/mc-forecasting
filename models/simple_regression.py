from datetime import timedelta, datetime
import numpy as np
import pandas as pd
from estimator import Estimator
from sklearn.linear_model import LinearRegression
from typing import Dict, Any
import optuna

class SimpleRegressionEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db)
        self.model = LinearRegression()
        self.optimization_frequency = timedelta(days=60)
        self.performance_threshold = np.inf

    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:

        X = np.hstack([
            data['generation_forecast'].values,
            data['load_forecast'].values,
            data['wind_solar_forecast'].values
        ])
        y = data['day_ahead_prices'].values

        return {'X': X, 'y': y}

    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        # LinearRegression doesn't need to split data
        # return {'train': train_data, 'valid': train_data}
        pass

    def fit(self, prepared_data: Dict[str, pd.DataFrame]):
        self.model.fit(prepared_data['X'], prepared_data['y'])

    def predict(self, prepared_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(prepared_data['X']))

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        # LinearRegression doesn't have hyperparameters to tune, but you could add some if needed
        return {}

    def optimize(self, train_data: Dict[str, pd.DataFrame], n_trials: int = 50, current_date: datetime = None):
        # Do nothing or log that optimization is not implemented for this estimator
        # print(f"Optimization not implemented for {self.name}")
        pass