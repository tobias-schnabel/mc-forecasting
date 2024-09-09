import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from datetime import timedelta, datetime
from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

from estimator import Estimator

# Suppress Optuna warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


class NBEATSxBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, stack_type):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, output_size)
        self.stack_type = stack_type

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast(h)
        forecast = self.forecast(h)
        return backcast, forecast


class NBEATSxModel(LightningModule):
    def __init__(self, input_size, output_size, hidden_size, stack_types, n_blocks, learning_rate):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSxBlock(input_size, output_size, hidden_size, stack_type)
            for _ in range(n_blocks)
            for stack_type in stack_types
        ])
        self.learning_rate = learning_rate

    def forward(self, x):
        residuals = x
        forecast = torch.zeros(x.size(0), self.blocks[-1].forecast.out_features, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class NBEATSxEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False, device='mps'):
        super().__init__(name, results_dir, use_db)
        self.model = None
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=7)
        self.performance_threshold = 0.1
        self.n_trials = 5
        self.device = torch.device(device)  # Always use CPU
        self.verbose = False

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        X = np.hstack([
            data['generation_forecast'].values,
            data['load_forecast'].values,
            data['wind_solar_forecast'].values,
            data['coal_gas_cal'].values
        ])
        y = data['day_ahead_prices'].values

        # Convert to float type
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        return {'X': X, 'y': y}

    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        split_point = int(len(train_data['day_ahead_prices']) * 0.8)
        train = {k: v.iloc[:split_point] for k, v in train_data.items()}
        valid = {k: v.iloc[split_point:] for k, v in train_data.items()}
        return {'train': train, 'valid': valid}

    def fit(self, prepared_data: Dict[str, torch.Tensor]):
        X, y = prepared_data['X'], prepared_data['y']

        # Split data into train and validation
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        train_dataloader = self._create_dataloader(X_train, y_train)
        val_dataloader = self._create_dataloader(X_val, y_val)

        # Move model to device
        self.model = self.model.to(self.device)

        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            patience=5,
            verbose=False,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=1_000,
            callbacks=[early_stop_callback],
            accelerator='auto',  # Explicitly set to CPU
            devices=1,
            enable_progress_bar=self.verbose,
            logger=self.verbose,
            enable_checkpointing=False,
        )

        try:
            trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            print("Attempting to continue with the next trial...")

    def predict(self, prepared_data: Dict[str, torch.Tensor]) -> pd.DataFrame:
        X = prepared_data['X'].to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()
        return pd.DataFrame(preds)

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'n_blocks': trial.suggest_int('n_blocks', 1, 5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'stack_types': trial.suggest_categorical('stack_types', [
                'trend_seasonality',
                'trend_seasonality_exogenous',
                'generic_generic'
            ])
        }

    def set_model_params(self, **params):
        input_size = self.prepared_data['X'].shape[1]
        output_size = self.prepared_data['y'].shape[1]
        stack_types = params['stack_types'].split('_')
        self.model = NBEATSxModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=params['hidden_size'],
            stack_types=stack_types,
            n_blocks=params['n_blocks'],
            learning_rate=params['learning_rate']
        ).to(self.device)

    def optimize(self, train_data: Dict[str, pd.DataFrame], current_date: datetime):
        self.prepared_data = self.prepare_data(train_data)
        super().optimize(train_data, current_date)

    def _create_dataloader(self, X, y, batch_size=32):
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Keep this at 0 to avoid multiprocessing issues
            pin_memory=False
        )

    def objective(self, trial):
        params = self.define_hyperparameter_space(trial)
        self.set_model_params(**params)
        prepared_train_data = self.prepare_data(self.train_data)
        try:
            self.fit(prepared_train_data)
            # Your existing validation code here
            # For example:
            # validation_loss = self.validate(prepared_val_data)
            # return validation_loss
        except Exception as e:
            print(f"An error occurred during the trial: {str(e)}")
            return float('inf')  # Return a large value to indicate a failed trial
