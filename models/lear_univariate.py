import warnings

from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import warnings
from datetime import timedelta
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from joblib import Parallel, delayed, cpu_count
from estimator import Estimator
from evaluation_utils import mse as comp_mse

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LEAREstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db, required_history=7)
        self.models = {country: [None for _ in range(24)] for country in range(12)}
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=14)
        self.min_opt_days = 16
        self.performance_threshold = 0.1
        self.n_trials = 2
        self.n_countries = 12
        self.countries = None
        self.n_country_specific = 13
        self.eval_metric = "custom_metric"

    def compute_custom_metric(self, y_true, y_pred):
        total_aic = 0
        n_hours, n_countries = y_true.shape
        for country in range(self.n_countries):
            for hour in range(24):
                k = np.sum(self.models[country][hour].coef_ != 0)
                mse = comp_mse(y_true[hour, country], y_pred[hour, country])
                aic = 2 * k + np.log(mse)
                total_aic += aic
        return total_aic / (n_hours * n_countries)

    def set_model_params(self, **params):
        for country in range(self.n_countries):
            self.models[country] = [Lasso(max_iter=2500) if model is None else model for model in self.models[country]]
            for hour in range(24):
                self.models[country][hour].set_params(alpha=params[f"alpha_{country}_{hour}"])

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        if self.countries is None:
            self.countries = data['day_ahead_prices'].columns.tolist()
        X, y = self._build_features(data)

        for country in range(self.n_countries):
            for hour in range(24):
                X_item = X[country][hour]
                y_item = y[country][hour]

                if not is_train:
                    X_item = X_item[-1:]  # For test data, only use the last hour
                    y_item = y_item[-1:]

                X[country][hour] = X_item
                y[country][hour] = y_item

        return {"X": X, "y": y}

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

    def fit_single_model(self, country, hour, X, y):
        if self.models[country][hour] is None:
            self.models[country][hour] = Lasso(max_iter=2500)
        self.models[country][hour].fit(X, y)
        return country, hour, self.models[country][hour]

    def fit(self, prepared_data: Dict[str, np.ndarray]):
        X, y = prepared_data['X'], prepared_data['y']

        n_jobs = max(1, cpu_count())
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.fit_single_model)(country, hour, X[country][hour], y[country][hour])
            for country in range(self.n_countries)
            for hour in range(24)
        )

        for country, hour, fitted_model in results:
            self.models[country][hour] = fitted_model

    def predict_single_hour(self, country, hour, X_hour):
        predictions_scaled = self.models[country][hour].predict(X_hour)
        return predictions_scaled

    def predict(self, prepared_data: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
        X = prepared_data["X"]

        n_jobs = max(1, cpu_count())
        predictions = Parallel(n_jobs=n_jobs)(
            delayed(self.predict_single_hour)(country, hour, X[country][hour])
            for country in range(self.n_countries)
            for hour in range(24)
        )

        predictions = np.array(predictions).reshape(24, self.n_countries)

        hours = [f'h{i:02d}' for i in range(24)]
        df = pd.DataFrame(predictions, index=hours, columns=self.countries)
        return df

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            f"alpha_{country}_{hour}": trial.suggest_float(f"alpha_{country}_{hour}", 1e-5, 10, log=True)
            for country in range(self.n_countries)
            for hour in range(24)
        }

    def _build_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        countries = data['day_ahead_prices'].columns
        n_countries = len(countries)

        country = countries[0]
        df_get_hours = pd.DataFrame(index=data['day_ahead_prices'].index)

        df_get_hours[f'price_lag_{168}'] = data['day_ahead_prices'][country].shift(168)
        df_get_hours = df_get_hours.dropna()
        n_hours = len(df_get_hours.index)

        common_features = pd.DataFrame({
            'coal': data['coal_gas_cal']['coal'],
            'gas': data['coal_gas_cal']['gas'],
            'day_of_week': data['coal_gas_cal']['day_of_week']
        }, index=data['day_ahead_prices'].index)

        hour_dummies = pd.get_dummies(data['coal_gas_cal']['hour'], prefix='hour', drop_first=True)
        common_features = pd.concat([common_features, hour_dummies], axis=1)[-n_hours:]

        coal_gas = common_features.iloc[:, :2]
        common_features = common_features.drop(common_features.columns[:2], axis=1)
        common_features = pd.concat([common_features, coal_gas], axis=1)

        n_features_with_dap = len(common_features.columns) + ((self.n_country_specific + 1) * n_countries) + 2
        n_features = len(common_features.columns) + (self.n_country_specific * n_countries) + 2

        X = np.zeros((n_hours * n_countries, n_features_with_dap))
        X[:, 1:len(common_features.columns) + 1] = np.tile(common_features.values, (n_countries, 1))

        df_shape = (n_hours * n_countries, len(common_features.columns))
        df_common_features = pd.DataFrame(np.zeros(df_shape), columns=common_features.columns)

        tiled_values = np.tile(common_features.values, (n_countries, 1))

        df_common_features.iloc[:, :] = tiled_values.astype(float)
        df_common_features = df_common_features.iloc[:, 1:24]

        dataframes = []

        for i in range(24):
            if i == 0:
                df_filtered = df_common_features[(df_common_features == False).all(axis=1)]
                df_filtered['hour_0'] = True
                columns_to_drop = df_filtered.columns[(df_filtered == False).all()]
                df_filtered.drop(columns=columns_to_drop, inplace=True)
            else:
                df_filtered = df_common_features[df_common_features.iloc[:, i - 1] == True]
                columns_to_drop = df_filtered.columns[(df_filtered == False).all()]
                df_filtered.drop(columns=columns_to_drop, inplace=True)

            dataframes.append(df_filtered)

        X_dict = {country: {} for country in range(n_countries)}
        y_dict = {country: {} for country in range(n_countries)}

        day_ahead_location = []
        feature_start = len(common_features.columns) + 1
        for i, country in enumerate(countries):
            country_features = pd.DataFrame(index=data['day_ahead_prices'].index)
            country_features[f'day_ahead_price_{country}'] = data['day_ahead_prices'][country]

            for lag in [24, 48, 72, 168]:
                country_features[f'price_lag_{lag}'] = data['day_ahead_prices'][country].shift(lag)

            country_features[f'generation_forecast_{country}'] = data['generation_forecast'][country]
            for lag in [24, 168]:
                country_features[f'generation_forecast_lag_{lag}'] = data['generation_forecast'][country].shift(lag)

            country_features[f'load_forecast_{country}'] = data['load_forecast'][country]
            for lag in [24, 168]:
                country_features[f'load_forecast_lag_{lag}'] = data['load_forecast'][country].shift(lag)

            country_features[f'wind_solar_forecast_{country}'] = data['wind_solar_forecast'][country]
            for lag in [24, 168]:
                country_features[f'wind_solar_forecast_lag_{lag}'] = data['wind_solar_forecast'][country].shift(lag)

            country_features = country_features.dropna()

            feature_end = feature_start + self.n_country_specific + 1
            X[i * n_hours:(i + 1) * n_hours, feature_start:feature_end] = country_features.values
            day_ahead_location.append(feature_start)
            feature_start += self.n_country_specific

            X[i * n_hours:(i + 1) * n_hours, 0] = i

        X_without_dap = np.delete(X.copy(), day_ahead_location, axis=1)
        X_only_dap = X[:, day_ahead_location]

        for country in range(n_countries):
            for hour in range(24):
                country_hour_indices = dataframes[hour].index[dataframes[hour].index % n_countries == country]
                X_dict[country][hour] = X_without_dap[country_hour_indices]
                y_dict[country][hour] = X_only_dap[country_hour_indices, country]

        return X_dict, y_dict
