import warnings
from datetime import timedelta
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from estimator import Estimator

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class LEAREstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db)
        self.model = None
        self.optimization_frequency = timedelta(days=30)
        self.performance_threshold = 0.1
        self.n_trials = 50
        self.n_countries = 12

    def set_model_params(self, **params):
        if self.model is None:
            self.model = Lasso(max_iter=2500)
        self.model.set_params(**params)

    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        X = self._build_features(data)
        y = data["day_ahead_prices"].values.flatten()
        return {"X": X, "y": y}

    def _build_features(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        countries = data['day_ahead_prices'].columns
        n_countries = len(countries)
        n_hours = len(data['day_ahead_prices'])

        # Initialize feature matrix
        n_features = 4 + 5 + 2 + n_countries  # lags + exogenous + time features + country indicators
        X = np.zeros((n_hours * n_countries, n_features))

        for i, country in enumerate(countries):
            df = pd.DataFrame(index=data['day_ahead_prices'].index)

            # Add price lags
            for lag in [24, 48, 72, 168]:  # 1 day, 2 days, 3 days, 1 week
                df[f'price_lag_{lag}'] = data['day_ahead_prices'][country].shift(lag)

            # Add exogenous variables
            df['generation_forecast'] = data['generation_forecast'][country]
            df['load_forecast'] = data['load_forecast'][country]
            df['wind_solar_forecast'] = data['wind_solar_forecast'][country]
            df['coal'] = data['coal_gas_cal']['coal']
            df['gas'] = data['coal_gas_cal']['gas']
            df['hour'] = data['coal_gas_cal']['hour']
            df['day_of_week'] = data['coal_gas_cal']['day_of_week']

            # Add country indicators
            for j, c in enumerate(countries):
                df[f'country_{c}'] = 1 if c == country else 0

            # Handle missing data
            df = df.fillna(0)  # Replace NaN with 0
            # df = df.dropna()
            # Add to the main feature matrix
            X[i * n_hours:(i + 1) * n_hours, :] = df.values

        return X

    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        # Extract the index of the split point
        split_index = train_data["day_ahead_prices"].index[int(len(train_data["day_ahead_prices"]) * 0.8)]

        # Find the nearest whole day index
        split_whole_day = split_index.normalize() + timedelta(hours=23)

        # Split the data at the nearest whole day index
        train = {k: v.loc[:split_whole_day] for k, v in train_data.items()}
        valid = {k: v.loc[split_whole_day + pd.Timedelta(hours=1):] for k, v in train_data.items()}
        return {"train": train, "valid": valid}

    def fit(self, prepared_data: Dict[str, np.ndarray]):
        X, y = prepared_data['X'], prepared_data['y']
        if self.model is None:
            self.model = Lasso(max_iter=2500)
        self.model.fit(X, y)

    def predict(self, prepared_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        X = prepared_data["X"]
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions.reshape(self.n_countries, -1).T)

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True)}