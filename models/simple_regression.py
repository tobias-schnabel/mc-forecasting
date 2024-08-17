from datetime import timedelta

import numpy as np
import pandas as pd
from estimator import Estimator
from sklearn.linear_model import LinearRegression
from typing import Dict, Any
import optuna

class SimpleRegressionEstimator(Estimator):
    def __init__(self, name: str, base_dir: str):
        super().__init__(name, base_dir)
        self.model = LinearRegression()
        self.optimization_frequency = timedelta(days=60)
        self.performance_threshold = 0.9

    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:

        X = np.hstack([
            data['generation_forecast'].values,
            data['load_forecast'].values,
            data['wind_solar_forecast'].values
        ])
        y = data['day_ahead_prices'].values

        return {'X': X, 'y': y}

    def fit(self, train_data: Dict[str, pd.DataFrame]):
        prepared_data = self.prepare_data(train_data)
        self.model.fit(prepared_data['X'], prepared_data['y'])

    def predict(self, test_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prepared_data = self.prepare_data(test_data)
        predictions = self.model.predict(prepared_data['X'])
        return pd.DataFrame(predictions, index=test_data['day_ahead_prices'].index,
                            columns=test_data['day_ahead_prices'].columns)

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        # LinearRegression doesn't have hyperparameters to tune, but you could add some if needed
        return {}

    def optimize(self, train_data: Dict[str, pd.DataFrame], valid_data: Dict[str, pd.DataFrame], n_trials: int = 50):
        # Do nothing or log that optimization is not implemented for this estimator
        # print(f"Optimization not implemented for {self.name}")
        pass