from datetime import timedelta
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd
from mcnnm.core_utils import normalize

from estimator import Estimator
from mcnnm import complete_matrix


@jax.jit
def normalize_3d(tensor):
    """
    Normalize each 2D slice of the input 3D tensor.
    Return the normalized tensor and the norms for each slice.
    """
    return jax.vmap(normalize)(tensor)


class MCNNMEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False):
        super().__init__(name, results_dir, use_db, required_history=0)
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=7)
        self.performance_threshold = 0.1
        self.n_trials = 1
        self.eval_metric = "MAE"
        self.training_data = None
        self.min_opt_days = 8
        self.hyperparameters = {
            "lambda_L": None,
            "lambda_H": None,
            "n_lambda": 20,
            "K": 6,
            "step_size": 4,
            "horizon": 1,
            "max_window_size": None,
            "max_iter": 10_000,
            "tol": 1e-4,
            "use_time_fe": True,
        }
        self.X = None,
        self.verbose = False

    def set_model_params(self, **params):
        self.hyperparameters.update(params)

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

            num_train_cols = (self.training_data['Y']).shape[1]

            num_rows = data['day_ahead_prices'].T.shape[0]
            num_test_cols = 24
            num_rows = data['day_ahead_prices'].T.shape[0]
            W_df = pd.DataFrame(np.zeros((num_rows, num_train_cols)))
            W_df.iloc[:, -num_test_cols:] = 1
            W = jnp.array(W_df.values)
            Y = jnp.array(self.training_data['Y'].values)
            Z = jnp.array(Z_df.values, dtype=jnp.float32)

            # Prep V tensor
            v_1 = self.training_data['V_1'].values
            v_2 = self.training_data['V_2'].values
            v_3 = self.training_data['V_3'].values
            V = jnp.stack([v_1, v_2, v_3], axis=2)
            # V_normalized, V_norms = normalize_3d(V)
            V_log = jnp.log1p(V)  # take log of V, method returns NaNs otherwise

            return {'Y': Y, 'Z': Z, 'V': V_log, 'W': W, 'num_test_cols': num_test_cols}
        else:
            num_train_cols = (self.training_data['Y']).shape[1]

            self.training_data['Y'] = pd.concat([self.training_data['Y'], data['day_ahead_prices'].T], axis=1).astype(
                float)
            self.training_data['Z'] = pd.concat([self.training_data['Z'], data['coal_gas_cal']], axis=0).astype(float)
            self.training_data['V_1'] = pd.concat([self.training_data['V_1'], data['load_forecast'].T], axis=1).astype(
                float)
            self.training_data['V_2'] = pd.concat([self.training_data['V_2'], data['generation_forecast'].T],
                                                  axis=1).astype(float)
            self.training_data['V_3'] = pd.concat([self.training_data['V_3'], data['wind_solar_forecast'].T],
                                                  axis=1).astype(float)

            num_test_cols = data['day_ahead_prices'].T.shape[1]
            num_rows = data['day_ahead_prices'].T.shape[0]
            W_df = pd.DataFrame(np.zeros((num_rows, (num_train_cols + num_test_cols))))
            W_df.iloc[:, -num_test_cols:] = 1
            W = jnp.array(W_df.values)
            Y = jnp.array(self.training_data['Y'].values)
            Z = jnp.array(self.training_data['Z'].values)

            # Prep V tensor
            v_1 = self.training_data['V_1'].values
            v_2 = self.training_data['V_2'].values
            v_3 = self.training_data['V_3'].values
            V = jnp.stack([v_1, v_2, v_3], axis=2)
            # V_normalized, V_norms = normalize_3d(V)
            V_log = jnp.log1p(V)  # take log of V, method returns NaNs otherwise

            self.training_data = None

            return {'Y': Y, 'Z': Z, 'V': V_log, 'W': W, 'num_test_cols': num_test_cols}

    def fit(self, prepared_data: Dict[str, Any]):
        pass

    def predict(self, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        # Data
        Y = prepared_data['Y']
        Z = prepared_data['Z']
        V = prepared_data['V']
        W = prepared_data['W']
        # X = jnp.array(self.X.values)

        # Parameters
        lambda_L = self.hyperparameters['lambda_L'] if self.hyperparameters['lambda_L'] is not None else None
        lambda_H = self.hyperparameters['lambda_H'] if self.hyperparameters['lambda_H'] is not None else None
        n_lambda = self.hyperparameters['n_lambda']
        K = self.hyperparameters['K']
        step_size = self.hyperparameters['step_size']
        horizon = self.hyperparameters['horizon']
        # max_window_size = self.hyperparameters['max_window_size'] if self.hyperparameters['max_window_size'] is not None else None
        use_time_fe = self.hyperparameters['use_time_fe']

        # Set initial window to be T - 24
        N, T = Y.shape
        initial_window = T - 24

        # Reduce max iterations and tolerance for faster optimization
        max_iter = self.hyperparameters['max_iter'] if self.hyperparameters['max_iter'] is not None else 10_000
        tol = self.hyperparameters['tol'] if self.hyperparameters['tol'] is not None else 1e-2

        if lambda_L is None and self.verbose:
            print(f"Optimizing lambda_L and lambda_H with {n_lambda} values, {max_iter} max iter, and {tol} tol")

        if self.verbose:
            print(f"Y dimensions: {Y.shape}")
        # Fit
        Y_completed, opt_lambda_L, opt_lambda_H = complete_matrix(
            Y=Y,
            Mask=W,
            X=None,
            Z=Z,
            V=V,
            Omega=None,
            use_unit_fe=True,
            use_time_fe=use_time_fe,
            lambda_L=lambda_L,
            lambda_H=lambda_H,
            n_lambda=n_lambda,
            validation_method="holdout",
            K=K,
            initial_window=initial_window,
            step_size=step_size,
            horizon=horizon,
            max_window_size=None,
            max_iter=max_iter,
            tol=tol
        )

        # Pass forward the lambdas between optimization steps
        self.hyperparameters['lambda_L'] = opt_lambda_L.item()
        self.hyperparameters['lambda_H'] = opt_lambda_H.item()

        # Reset max_iter and tol
        self.hyperparameters['max_iter'] = None  # type: ignore
        self.hyperparameters['tol'] = None  # type: ignore

        preds = pd.DataFrame(Y_completed[:, -prepared_data['num_test_cols']:]).T
        return pd.DataFrame(preds)

    def split_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        return {'train': data, 'valid': data}

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:

        return {
            "use_time_fe": trial.suggest_categorical("use_time_fe", [True, False]),
            "lambda_L": None,  # reset lambda_L to trigger validation
            "lambda_H": None,  # reset lambda_H
            "n_lambda": 18,
            # "max_window_size": trial.suggest_int("max_window_size", 336, 672, step=4),  # take last 2 weeks' worth of data
            "K": 3,  # 6 folds
            "step_size": 8,  # Move val window forward by 8 hours at a time
            "horizon": 1,  # Predict 1 hour ahead
            "max_iter": 1_000,  # reduce max_iter for faster optimization
            "tol": 1.0  # increase tolerance for faster optimization
        }
