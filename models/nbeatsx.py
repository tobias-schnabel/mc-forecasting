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
from evaluation_utils import calculate_opt_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class PanelNBEATSxBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast(h)
        forecast = self.forecast(h)
        return backcast, forecast


class PanelNBEATSxModel(LightningModule):
    def __init__(self, input_size, output_size, hidden_size, n_blocks, n_countries, learning_rate):
        super().__init__()
        self.country_embedding = nn.Embedding(n_countries, 8)  # 8-dimensional embedding for countries
        self.blocks = nn.ModuleList([
            PanelNBEATSxBlock(input_size + 8, output_size, hidden_size)
            for _ in range(n_blocks)
        ])
        self.learning_rate = learning_rate

    def forward(self, x, country_idx):
        country_emb = self.country_embedding(country_idx)
        x = torch.cat([x, country_emb], dim=-1)

        residuals = x
        forecast = torch.zeros(x.size(0), self.blocks[-1].forecast.out_features, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

    def training_step(self, batch, batch_idx):
        x, y, country_idx = batch
        y_hat = self(x, country_idx)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, country_idx = batch
        y_hat = self(x, country_idx)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class PanelNBEATSxEstimator(Estimator):
    def __init__(self, name: str, results_dir: str, use_db: bool = False, device='mps'):
        super().__init__(name, results_dir, use_db)
        self.model = None
        self.optimization_frequency = timedelta(days=30)
        self.optimization_wait = timedelta(days=7)
        self.performance_threshold = 0.1
        self.n_trials = 5
        self.device = torch.device(device)
        self.verbose = False
        self.n_countries = 12  # Assuming 12 countries as mentioned earlier

    def prepare_data(self, data: Dict[str, pd.DataFrame], is_train: bool = True) -> Dict[str, Any]:
        countries = data['day_ahead_prices'].columns
        X_list, y_list, country_idx_list = [], [], []

        for i, country in enumerate(countries):
            X_country = np.column_stack([
                data['generation_forecast'][country].values,
                data['load_forecast'][country].values,
                data['wind_solar_forecast'][country].values,
                data['coal_gas_cal'].values
            ])
            y_country = data['day_ahead_prices'][country].values

            X_list.append(X_country)
            y_list.append(y_country)
            country_idx_list.append(np.full(len(y_country), i))

        X = np.vstack(X_list).astype(np.float32)
        y = np.concatenate(y_list).astype(np.float32)
        country_idx = np.concatenate(country_idx_list).astype(np.int64)

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(-1)  # Add a dimension for output size
        country_idx = torch.LongTensor(country_idx)

        return {'X': X, 'y': y, 'country_idx': country_idx, 'day_ahead_prices': data['day_ahead_prices']}

    def split_data(self, train_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        split_point = int(len(train_data['day_ahead_prices']) * 0.8)
        train = {k: v.iloc[:split_point] for k, v in train_data.items()}
        valid = {k: v.iloc[split_point:] for k, v in train_data.items()}
        return {'train': train, 'valid': valid}

    def fit(self, prepared_data: Dict[str, torch.Tensor]):
        X, y, country_idx = prepared_data['X'], prepared_data['y'], prepared_data['country_idx']

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        country_idx_train, country_idx_val = country_idx[:train_size], country_idx[train_size:]

        train_dataloader = self._create_dataloader(X_train, y_train, country_idx_train)
        val_dataloader = self._create_dataloader(X_val, y_val, country_idx_val)

        self.model = self.model.to(self.device)

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=False,
            mode='min'
        )

        trainer = Trainer(
            max_epochs=100,
            callbacks=[early_stop_callback],
            accelerator='auto',
            devices=1,
            enable_progress_bar=self.verbose,
            logger=self.verbose,
            enable_checkpointing=False,
            num_nodes=1
        )

        try:
            trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            print("Attempting to continue with the next trial...")

    def predict(self, prepared_data: Dict[str, torch.Tensor]) -> pd.DataFrame:
        X, country_idx = prepared_data['X'].to(self.device), prepared_data['country_idx'].to(self.device)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X, country_idx).cpu().numpy()

        # Reshape predictions to match the original data structure
        preds = preds.reshape(-1, self.n_countries)
        return pd.DataFrame(preds, columns=prepared_data['day_ahead_prices'].columns)

    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'n_blocks': trial.suggest_int('n_blocks', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        }

    def set_model_params(self, **params):
        input_size = self.prepared_data['X'].shape[1]
        output_size = self.prepared_data['y'].shape[1]
        self.model = PanelNBEATSxModel(
            input_size=input_size,
            output_size=output_size,
            hidden_size=params['hidden_size'],
            n_blocks=params['n_blocks'],
            n_countries=self.n_countries,
            learning_rate=params['learning_rate']
        ).to(self.device)

    def optimize(self, train_data: Dict[str, pd.DataFrame], current_date: datetime):
        self.prepared_data = self.prepare_data(train_data)
        split_data = self.split_data(train_data)
        self.train_data = split_data['train']
        self.valid_data = split_data['valid']
        super().optimize(train_data, current_date)

    def _create_dataloader(self, X, y, country_idx, batch_size=32):
        dataset = torch.utils.data.TensorDataset(X, y, country_idx)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def objective(self, trial):
        params = self.define_hyperparameter_space(trial)
        self.set_model_params(**params)
        prepared_train_data = self.prepare_data(self.train_data)
        try:
            self.fit(prepared_train_data)

            # Validate on the validation set
            prepared_val_data = self.prepare_data(self.valid_data)
            predictions = self.predict(prepared_val_data)
            actuals = self.valid_data['day_ahead_prices'].values

            metrics = calculate_opt_metrics(predictions.values, actuals)
            return metrics[self.eval_metric]
        except Exception as e:
            print(f"An error occurred during the trial: {str(e)}")
            return float('inf')  # Return a large value to indicate a failed trial