import warnings
from datetime import timedelta
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from data_utils import InvariantScaler
from estimator import Estimator
from evaluation_utils import mse as comp_mse

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
        self.n_trials = 20
        self.n_countries = 12
        self.n_country_specific = 13
        self.scaler_X = InvariantScaler()
        self.scaler_y = InvariantScaler()
        self.eval_metric = "custom_metric"

    def compute_custom_metric(self, y_true, y_pred):
        n = len(y_true)
        k = np.sum(self.model.coef_ != 0)  # Number of non-zero coefficients
        mse = comp_mse(y_true, y_pred)
        aic = 2 * k + n * np.log(mse)
        return aic

    def set_model_params(self, **params):
        if self.model is None:
            self.model = Lasso(max_iter=2500)
        self.model.set_params(**params)

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        X = self._build_features(data)
        n = len(X)
        y = data["day_ahead_prices"].values.flatten()[-n:]

        # Separate features and dummy variables
        X_features = X[:, 25:]  # Exclude the first 25 columns which are dummies
        X_dummies = X[:, :25]  # Keep the first 25 columns which are dummies

        if is_train:
            # Fit and transform X (excluding dummies) and y
            X_features_scaled = self.scaler_X.fit_transform(X_features)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            # For test data, we need to scale each step independently
            X_features_scaled = np.zeros_like(X_features)
            y_scaled = np.zeros_like(y)
            for i in range(len(X_features)):
                X_features_scaled[i] = self.scaler_X.transform(X_features[i].reshape(1, -1))
                y_scaled[i] = self.scaler_y.transform(y[i].reshape(1, -1))

        # Recombine scaled features with unscaled dummies
        X_scaled = np.hstack((X_features_scaled, X_dummies))

        if not is_train:
            X_scaled = X_scaled[-24 * self.n_countries:, :]  # For test data, only use the last 24 hours
            y_scaled = y_scaled[-24:]

        return {"X": X_scaled, "y": y_scaled}

    def _build_features(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        countries = data['day_ahead_prices'].columns
        n_countries = len(countries)

        # Get the number of non-nan columns in the final df by adding max lag to one and droppin NaNs
        country = countries[0]
        df_get_hours = pd.DataFrame(index=data['day_ahead_prices'].index)

        # Add max price lag
        df_get_hours[f'price_lag_{168}'] = data['day_ahead_prices'][country].shift(168)
        df_get_hours = df_get_hours.dropna()
        n_hours = len(df_get_hours.index)

        # Create common features
        # Prepare common features (coal, gas, day_of_week, hour dummies)
        common_features = pd.DataFrame({
            'coal': data['coal_gas_cal']['coal'],
            'gas': data['coal_gas_cal']['gas'],
            'day_of_week': data['coal_gas_cal']['day_of_week']
        }, index=data['day_ahead_prices'].index)

        # Create hour dummies
        hour_dummies = pd.get_dummies(data['coal_gas_cal']['hour'], prefix='hour', drop_first=True)
        common_features = pd.concat([common_features, hour_dummies], axis=1)[-n_hours:]  # drop na hours

        # Extract coal and gas columns, drop, and append to the end
        coal_gas = common_features.iloc[:, :2]
        common_features = common_features.drop(common_features.columns[:2], axis=1)
        common_features = pd.concat([common_features, coal_gas], axis=1)

        n_features = len(common_features.columns) + (
                self.n_country_specific * n_countries) + 2
        # Initialize feature matrix
        X = np.zeros((n_hours * n_countries, n_features))

        # Add common features to X (these will be the same for all countries)
        X[:, 1:len(common_features.columns) + 1] = np.tile(common_features.values, (n_countries, 1))

        feature_start = len(common_features.columns) + 1
        for i, country in enumerate(countries):
            # Country-specific features
            country_features = pd.DataFrame(index=data['day_ahead_prices'].index)

            # Add current day-ahead price
            # country_features[f'day_ahead_price_{country}'] = data['day_ahead_prices'][country] No, this is dep var

            # Add price lags
            for lag in [24, 48, 72, 168]:
                country_features[f'price_lag_{lag}'] = data['day_ahead_prices'][country].shift(lag)

            # Add exogenous variables and lags
            country_features[f'generation_forecast_{country}'] = data['generation_forecast'][country]
            for lag in [24, 168]:
                country_features[f'generation_forecast_lag_{lag}'] = data['generation_forecast'][country].shift(lag)

            country_features[f'load_forecast_{country}'] = data['load_forecast'][country]
            for lag in [24, 168]:
                country_features[f'load_forecast_lag_{lag}'] = data['load_forecast'][country].shift(lag)

            country_features[f'wind_solar_forecast_{country}'] = data['wind_solar_forecast'][country]
            for lag in [24, 168]:
                country_features[f'wind_solar_forecast_lag_{lag}'] = data['wind_solar_forecast'][country].shift(lag)

            country_features = country_features.dropna()  # Drop rows that don't have lagged values

            # Add country-specific features to X
            feature_end = feature_start + self.n_country_specific
            X[i * n_hours:(i + 1) * n_hours, feature_start:feature_end] = country_features.values
            feature_start += self.n_country_specific  # Update feature start index because we want wide data

            # Add country as a categorical variable
            X[i * n_hours:(i + 1) * n_hours, 0] = i  # Use integer encoding for the country

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
        split_index = data['day_ahead_prices'].index[split_point]  # Get the index at the split point
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
        predictions_scaled = self.model.predict(X)

        # Inverse transform the predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        return pd.DataFrame(predictions.reshape(self.n_countries, -1).T)

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {"alpha": trial.suggest_float("alpha", 1e-5, 10, log=True)}
