from datetime import timedelta
from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd
from mcnnm import complete_matrix

from estimator import Estimator


class MCNNMEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db, required_history=0)
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=3)
        self.performance_threshold = 0.1
        self.n_trials = 10
        self.eval_metric = "MAE"
        self.training_data = None
        self.min_opt_days = 16
        self.hyperparameters = {
            "lambda_L": None,
            "lambda_H": None,
            "n_lambda": 10,
            "K": 5,
            "step_size": 10,
            "horizon": 24
        }
        self.X = None

    def set_model_params(self, **params):
        self.hyperparameters = params

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        # Outcome (Y)
        Y_df = pd.DataFrame(data['day_ahead_prices']).T

        # Time-specific covariates (Z)
        Z_df = pd.DataFrame(data['coal_gas_cal'])

        # Unit-specific covariates (X): None for now
        # Unit-time specific covariates (V)
        load_df = pd.DataFrame(data['load_forecast']).T
        gen_df = pd.DataFrame(data['generation_forecast']).T
        ws_df = pd.DataFrame(data['wind_solar_forecast']).T

        if is_train:
            datadict = {
                "Y": Y_df,
                "Z": Z_df,
                "V_1": load_df,
                "V_2": gen_df,
                "V_3": ws_df
            }
            self.training_data = datadict
        else:
            num_train_cols = (self.training_data['Y']).shape[1]
            num_test_cols = (data['day_ahead_prices'].T).shape[1]
            self.training_data['Y'] = pd.concat([self.training_data['Y'], data['day_ahead_prices'].T], axis=1).astype(
                float)
            self.training_data['Z'] = pd.concat([self.training_data['Z'], data['coal_gas_cal']], axis=0).astype(float)
            self.training_data['V_1'] = pd.concat([self.training_data['V_1'], data['load_forecast'].T], axis=1).astype(
                float)
            self.training_data['V_2'] = pd.concat([self.training_data['V_2'], data['generation_forecast'].T],
                                                  axis=1).astype(float)
            self.training_data['V_3'] = pd.concat([self.training_data['V_3'], data['wind_solar_forecast'].T],
                                                  axis=1).astype(float)

            num_test_cols = (data['day_ahead_prices'].T).shape[1]
            num_rows = (data['day_ahead_prices'].T).shape[0]
            W_df = pd.DataFrame(np.zeros((num_rows, (num_train_cols + num_test_cols))))
            W_df.iloc[:, -num_test_cols:] = 1
            W = jnp.array(W_df.values)
            Y = jnp.array(self.training_data['Y'].values)
            Z = jnp.array(self.training_data['Z'].values)

            # Prep V tensor
            v_1 = jnp.array(self.training_data['V_1'].values)
            v_2 = jnp.array(self.training_data['V_2'].values)
            v_3 = jnp.array(self.training_data['V_3'].values)
            V = jnp.stack([v_1, v_2, v_3], axis=2)
            self.training_data = None

            return {'Y': Y, 'Z': Z, 'V': V, 'W': W, 'num_test_cols': num_test_cols}

    def fit(self, prepared_data: Dict[str, Any]):
        pass

    def predict(self, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        Y = prepared_data['Y']
        Z = prepared_data['Z']
        V = prepared_data['V']
        W = prepared_data['W']
        X = jnp.array(self.X.values)
        lambda_L = self.hyperparameters['lambda_L']
        lambda_H = self.hyperparameters['lambda_H']
        n_lambda = self.hyperparameters['n_lambda']
        K = self.hyperparameters['K']
        step_size = self.hyperparameters['step_size']
        horizon = self.hyperparameters['horizon']
        Y_completed, opt_lambda_L, opt_lambda_H = complete_matrix(
            Y=Y,
            Mask=W,
            X=X,
            Z=Z,
            V=V,
            Omega=None,
            use_unit_fe=True,
            use_time_fe=True,
            lambda_L=lambda_L,
            lambda_H=lambda_H,
            n_lambda=n_lambda,
            validation_method="holdout",
            K=K,
            initial_window=24,
            step_size=step_size,
            horizon=horizon,
            max_window_size=None,
        )

        # Pass forward the lambdas between optimization steps
        self.hyperparameters['lambda_L'] = opt_lambda_L
        self.hyperparameters['lambda_H'] = opt_lambda_H
        preds = pd.DataFrame(Y_completed[:, prepared_data['num_test_cols']:]).T
        return pd.DataFrame(preds)

    def split_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        return {'train': data, 'valid': data}

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.hyperparameters['lambda_L'] is not None:
            lambda_L = self.hyperparameters['lambda_L']
        if self.hyperparameters['lambda_H'] is not None:
            lambda_H = self.hyperparameters['lambda_H']

        return {
            "use_unit_fe": trial.suggest_categorical("use_unit_fe", [True, False]),
            "use_time_fe": trial.suggest_categorical("use_time_fe", [True, False]),
            "lambda_L": None,  # reset lambda_L
            "n_lambda_L": trial.suggest_int("n_lambda_L", 5, 20),
            "lambda_H": None,  # reset lambda_H
            "n_lambda_H": trial.suggest_int("n_lambda_H", 5, 20),
            "K": trial.suggest_int("K", 3, 10),
            "step_size": trial.suggest_int("step_size", 4, 100),
            "horizon": trial.suggest_int("horizon", 1, 24)
        }
