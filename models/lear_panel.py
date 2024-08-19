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
        super().__init__(name, results_dir, use_db, required_history=7)
        self.model = None
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=3)
        self.min_opt_days = 16
        self.performance_threshold = 0.1
        self.n_trials = 50
        self.n_countries = 12

    def set_model_params(self, **params):
        if self.model is None:
            self.model = Lasso(max_iter=2500)
        self.model.set_params(**params)

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        X = self._build_features(data)
        n = len(X)
        y = data["day_ahead_prices"].values.flatten()[-n:]

        if not is_train:
            X = X[-24 * self.n_countries:, :]  # For test data, only use the last 24 hours
            y = y[-24:]

        return {"X": X, "y": y}

    def _build_features(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Builds the feature matrix for the model from the provided data.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary containing the dataframes for day-ahead prices,
                                            generation forecast, load forecast, wind and solar forecast,
                                            and coal and gas calibration.

        Returns:
            np.ndarray: A 2D numpy array representing the feature matrix.
        """
        countries = data['day_ahead_prices'].columns
        n_countries = len(countries)
        country = countries[0]
        df_get_hours = pd.DataFrame(index=data['day_ahead_prices'].index)

        # Add max price lag
        df_get_hours[f'price_lag_{168}'] = data['day_ahead_prices'][country].shift(168)
        df_get_hours = df_get_hours.dropna()
        n_hours = len(df_get_hours.index)

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
            df = df.dropna()

            # Add to the main feature matrix
            X[i * n_hours:(i + 1) * n_hours, :] = df.values

        return X

    def split_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        # Ensure we have at least 16 days of data (8 for training, 8 for validation)
        min_required_days = 16
        if len(data['day_ahead_prices']) < min_required_days * 24:
            raise ValueError(f"Not enough data for optimization. Need at least {min_required_days} days.")

        # Calculate the number of hours for 8 days
        validation_hours = 8 * 24

        # Calculate the split point
        total_hours = len(data['day_ahead_prices'])
        split_point = total_hours - validation_hours

        # Ensure the split point is at least 80% of the data
        min_split_point = int(total_hours * 0.8)
        split_point = min(split_point, min_split_point)
        split_index = data['day_ahead_prices'].index[split_point] # Get the index at the split point
        split_whole_day = split_index.normalize()  # Find the nearest whole day index

        # Perform the split with adjusted validation set
        train_data = {k: v[v.index < split_whole_day] for k, v in data.items()}
        valid_data = {k: v[v.index >= split_whole_day] for k, v in data.items()}

        return {"train": train_data, "valid": valid_data}

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