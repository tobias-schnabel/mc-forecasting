import jax
import pandas as pd

jax.config.update("jax_enable_x64", False)  # set 32-bit precision for all floats

from data_utils import setup_environment
from models.mcnnm_32_bit import MCNNMEstimator

setup_paths, DataLoader, ForecastEngine = setup_environment()

data_dir, base_dir, tuning_dir, results_dir = setup_paths()
data_loader = DataLoader(data_dir)

mc = MCNNMEstimator("MCNNM-112_32", results_dir, use_db=False)

engine = ForecastEngine(data_loader, [mc])
engine.max_train_window = 112

start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2024-06-30", tz='UTC')

try:
    engine.run_forecast(start_date, end_date)
except KeyboardInterrupt:
    print("Training interrupted. Saving final results...")
    engine._save_final_results()
    engine._save_final_hyperparameters()
