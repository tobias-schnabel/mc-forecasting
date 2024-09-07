import pandas as pd

from data_utils import setup_environment
from models.lear_panel import LEAREstimator

setup_paths, DataLoader, ForecastEngine = setup_environment()

data_dir, base_dir, tuning_dir, results_dir = setup_paths()
data_loader = DataLoader(data_dir)

lear = LEAREstimator("LEAR-Panel-56", results_dir, use_db=False)

engine = ForecastEngine(data_loader, [lear])
engine.max_train_window = 56

start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2024-06-30", tz='UTC')

try:
    engine.run_forecast(start_date, end_date)
except KeyboardInterrupt:
    print("Training interrupted. Saving final results...")
    engine._save_final_results()
    engine._save_final_hyperparameters()
