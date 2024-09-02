import warnings
from datetime import timedelta, datetime
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from estimator import Estimator

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ElasticNetEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db)
        self.model = ElasticNet(random_state=42)
        self.optimization_frequency = timedelta(days=30)
        self.performance_threshold = 0.1
        self.n_trials = 10

    def compute_custom_metric(self, y_true, y_pred):
        return None

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        X = np.hstack([
            data['generation_forecast'].values,
            data['load_forecast'].values,
            data['wind_solar_forecast'].values,
            data['coal_gas_cal'].values
        ])
        y = data['day_ahead_prices'].values
        return {'X': X, 'y': y}

    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        split_point = int(len(train_data['day_ahead_prices']) * 0.8)
        train = {k: v.iloc[:split_point] for k, v in train_data.items()}
        valid = {k: v.iloc[split_point:] for k, v in train_data.items()}
        return {'train': train, 'valid': valid}

    def fit(self, prepared_data: Dict[str, np.ndarray]):
        self.model.fit(prepared_data['X'], prepared_data['y'])

    def predict(self, prepared_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame(self.model.predict(prepared_data['X']))

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)
        }

    def set_model_params(self, **params):
        self.model.set_params(**params)

    def optimize(self, train_data: Dict[str, pd.DataFrame], current_date: datetime):
        super().optimize(train_data, current_date)
