import datetime
import warnings
from datetime import timedelta
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from evaluation_utils import calculate_opt_metrics
from estimator import Estimator
from evaluation_utils import mse as comp_mse

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class LEAREstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db, required_history=7)
        self.models = [None for _ in range(24)]
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=2)
        self.min_opt_days = 16
        self.performance_threshold = 0.1
        self.n_trials = 1
        self.n_countries = 12
        self.countries = None
        self.n_country_specific = 13
        self.eval_metric = "custom_metric"

    def compute_custom_metric(self, y_true, y_pred):
        total_aic = 0
        n_hours, n_countries = y_true.shape
        for hour in range(24):
            # n = n_countries
            k = np.sum(self.models[hour].coef_ != 0)  # Number of non-zero coefficients
            mse = comp_mse(y_true[hour, :], y_pred[hour, :])
            aic = 2 * k + self.n_countries * np.log(mse)
            total_aic += aic
        return total_aic / n_hours  # Average AIC across all hours

    def set_model_params(self, **params):
        self.models = [Lasso(max_iter=2500) if model is None else model for model in self.models]
        if len(params) == 1:  # TODO: figure out what the hell is going wrong here
            for hour in range(24):
                self.models[hour].set_params(alpha=params.items())
        else:
            for hour in range(24):
                self.models[hour].set_params(alpha=params[f"alpha_{hour}"])

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        if self.countries is None:
            self.countries = data['day_ahead_prices'].columns.tolist()
        X, y = self._build_features(data)

        for key in X:
            X_item = X[key]
            y_item = y[key]

            if not is_train:
                X_item = X_item[-24 * self.n_countries:, :]  # For test data, only use the last 24 hours
                y_item = y_item[-24:]

            X[key] = X_item
            y[key] = y_item
        if not is_train and X[0].shape[0] != 12:
            print(f"X[0] shape: {X[0].shape}, {is_train}")
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

    def fit(self, prepared_data: Dict[str, np.ndarray]):
        X, y = prepared_data['X'], prepared_data['y']
        for i in range(24):
            if self.models[i] is None:
                self.models[i] = Lasso(max_iter=2500)
            self.models[i].fit(X[i], y[i])

    # def predict(self, prepared_data: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    #     X = prepared_data["X"]
    #     predictions = []
    #     for hour in range(24):
    #         predictions_scaled = self.models[hour].predict(X[hour])
    #         predictions.append(self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten())
    #
    #     # Stack the predictions horizontally (each row is a country, each column is an hour)
    #     predictions = np.column_stack(predictions)
    #
    #     # Create column labels for the 24 hours
    #     hours = [f'h{i:02d}' for i in range(24)]
    #
    #     # Create the DataFrame with countries as index and hours as columns
    #     df = pd.DataFrame(predictions, index=self.countries, columns=hours)
    #     df = df.transpose()
    #     return df

    # TRY
    # def predict(self, prepared_data: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    #     X = prepared_data["X"]
    #     y = prepared_data["y"]
    #     predictions = []
    #     for hour in range(24):
    #         predictions_scaled = self.models[hour].predict(X[hour])
    #         # predictions.append(self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten())
    #         predictions_hour = predictions_scaled
    #         predictions.append(predictions_hour)
    #
    #     # Stack the predictions horizontally (each row is an hour, each column is a country)
    #         # Stack the predictions vertically (each row is an hour)
    #     predictions = np.vstack(predictions)
    #
    #     # Ensure that we have the correct number of countries
    #     # if predictions.shape[1] != len(self.countries):
    #     #     predictions = predictions[:, :len(self.countries)]
    #     # predictions = predictions[-len(y),:]
    #     a = len(y.items())
    #     b, c = predictions.shape
    #
    #     # Create column labels for the 24 hours
    #     hours = [f'h{i:02d}' for i in range(24)]
    #
    #     # Create the DataFrame with hours as index and countries as columns
    #     # df = pd.DataFrame(predictions, index=hours, columns=self.countries)
    #     df = pd.DataFrame(predictions)
    #     # df = pd.DataFrame(predictions, columns=self.countries)
    #     return df

    # WORKING
    def predict(self, prepared_data: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
        X = prepared_data["X"]
        predictions = []
        for hour in range(24):
            X_hour = X[hour]
            predictions_scaled = self.models[hour].predict(X_hour)
            # print(f"Predictions shape before inverse_transform: {predictions_scaled.shape}")
            # predictions_hour = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            predictions_hour = predictions_scaled

            # Ensure we only keep the first n_countries predictions
            predictions_hour = predictions_hour[:self.n_countries]

            predictions.append(predictions_hour)

        predictions = np.array(predictions)

        hours = [f'h{i:02d}' for i in range(24)]
        df = pd.DataFrame(predictions, index=hours, columns=self.countries)
        return df

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            f"alpha_{hour}": trial.suggest_float(f"alpha_{hour}", 1e-5, 10, log=True)
            for hour in range(24)
        }

    # override base class method to allow for optimization of 24 separate models
    def optimize(self, train_data: Dict[str, pd.DataFrame], current_date: datetime):
        storage = self.manager.get_storage() if self.manager.use_db else None
        study_name = f"{self.name}_optimization_{current_date.strftime('%Y-%m-%d')}"

        split_data = self.split_data(train_data)
        train_subset = split_data['train']
        valid_subset = split_data['valid']
        prepared_train_data = self.prepare_data(train_subset, is_train=True)
        prepared_valid_data = self.prepare_data(valid_subset,
                                                is_train=True)  # changing this to false will send it all to hell

        def objective(trial):
            params = self.define_hyperparameter_space(trial)
            self.hyperparameters = params
            self.set_model_params(**params)  # Set model parameters
            self.fit(prepared_train_data)
            predictions = self.predict(prepared_valid_data)
            len_predictions = len(predictions)
            actuals = valid_subset['day_ahead_prices'].values[-len_predictions:]
            preds = predictions.values

            metrics = calculate_opt_metrics(preds, actuals)
            # Compute custom metric if implemented
            custom_metric = self.compute_custom_metric(actuals, preds)
            if custom_metric is not None:
                metrics['custom_metric'] = custom_metric

            return metrics[self.eval_metric]

        if self.manager.use_db:
            study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name,
                                        load_if_exists=True)
        else:
            study = optuna.create_study(direction="minimize", study_name=study_name)

        study.optimize(objective, n_trials=self.n_trials)

        # self.set_model_params(**study.best_params)
        self.last_optimization_date = current_date
        if study.best_params:
            self.set_model_params(**study.best_params)
            self.last_optimization_date = current_date
        else:
            print("Optimization failed to produce best parameters. Using default parameters.")
            default_params = {f"alpha_{hour}": 1e-3 for hour in range(24)}  # Example default value
            self.set_model_params(**default_params)


    def _build_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        # Initialize a dictionary to hold feature matrices for each hour
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

        # Calculate number of features
        # n_country_specific = 14  # current price + 4 price lags + 3*2 exogenous lags + 3 current exogenous

        n_features_with_dap = len(common_features.columns) + (
                (self.n_country_specific + 1) * n_countries) + 2
        n_features = len(common_features.columns) + (
                self.n_country_specific * n_countries) + 2
        # Initialize feature matrix
        X = np.zeros((n_hours * n_countries, n_features_with_dap))
        # X_without_dap = np.zeros((n_hours * n_countries, n_features))
        # Add common features to X (these will be the same for all countries)
        X[:, 1:len(common_features.columns) + 1] = np.tile(common_features.values, (n_countries, 1))
        # X_without_dap[:, 1:len(common_features.columns) + 1] = np.tile(common_features.values, (n_countries, 1))
        # Create a new DataFrame with the same shape as the target array slice
        df_shape = (n_hours * n_countries, len(common_features.columns))
        df_common_features = pd.DataFrame(np.zeros(df_shape), columns=common_features.columns)

        # Use np.tile to repeat the values of common_features for each country
        tiled_values = np.tile(common_features.values, (n_countries, 1))

        # Assign the tiled values to the new DataFrame
        df_common_features.iloc[:, :] = tiled_values.astype(float)
        df_common_features = df_common_features.iloc[:, 1:24]
        # Initialize an empty list to store the DataFrames
        dataframes = []

        # Iterate over the range of 24
        for i in range(24):
            if i == 0:
                # For the first DataFrame, filter rows where all columns are False
                df_filtered = df_common_features[(df_common_features == False).all(axis=1)]
                # Add a new column named 'hour_0' and set it to True for all rows
                df_filtered['hour_0'] = True
                columns_to_drop = df_filtered.columns[(df_filtered == False).all()]
                df_filtered.drop(columns=columns_to_drop, inplace=True)
            else:
                # For subsequent DataFrames, filter rows where the corresponding column is True
                df_filtered = df_common_features[df_common_features.iloc[:, i - 1] == True]
                columns_to_drop = df_filtered.columns[(df_filtered == False).all()]
                df_filtered.drop(columns=columns_to_drop, inplace=True)

            # Append the filtered DataFrame to the list
            dataframes.append(df_filtered)
        # Initialize an empty dictionary to store the numpy arrays
        X_dict = {}
        y_dict = {}
        # Iterate over the range of 24
        for i in range(24):
            # Get the number of rows for the current DataFrame
            n_rows = len(dataframes[i])

            # Create a numpy array with n_rows and n_features
            X_array = np.zeros((n_rows, n_features))
            y_array = np.zeros((n_rows,))
            # Assign the numpy array to the dictionary with the current index as the key
            X_dict[i] = X_array
            y_dict[i] = y_array

        day_ahead_location = []
        feature_start = len(common_features.columns) + 1
        for i, country in enumerate(countries):
            # Country-specific features
            country_features = pd.DataFrame(index=data['day_ahead_prices'].index)
            # Add current day-ahead price
            country_features[f'day_ahead_price_{country}'] = data['day_ahead_prices'][country]

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
            feature_end = feature_start + self.n_country_specific + 1
            X[i * n_hours:(i + 1) * n_hours, feature_start:feature_end] = country_features.values
            day_ahead_location.append(feature_start)
            feature_start += self.n_country_specific  # Update feature start index because we want wide data

            # Add country as a categorical variable
            X[i * n_hours:(i + 1) * n_hours, 0] = i  # Use integer encoding for the country

        X_without_dap = np.delete(X.copy(), day_ahead_location, axis=1)
        X_only_dap = X[:, day_ahead_location]

        for i in range(24):
            # Get the indices of the current DataFrame
            indices = dataframes[i].index

            # Select the corresponding rows from X
            selected_rows_X = X_without_dap[indices, :]
            selected_rows_y = X_only_dap[indices, :]
            n_y_rows = selected_rows_y.shape[0]
            # selected_y = selected_rows_y[:, :n_y_rows]

            y_needed = int(selected_rows_y.shape[0] / self.n_countries)
            non_zero_values = []
            for col in range(selected_rows_y.shape[1]):
                # Extract non-zero values from the column
                non_zero_col_values = selected_rows_y[:, col][selected_rows_y[:, col] != 0]

                # Take the last `n` non-zero values (use slicing to handle cases with fewer than `n` non-zero values)
                last_n_non_zero = non_zero_col_values[-y_needed:]

                # Append to the list
                non_zero_values.extend(last_n_non_zero)

            # Convert the list to a numpy array
            selected_y = np.array(non_zero_values)

            # Copy the selected rows into the corresponding array in X_dict
            X_dict[i][:] = selected_rows_X
            # y_dict[i][:] = np.diag(selected_y)
            y_dict[i][:] = selected_y

        return X_dict, y_dict
