import pandas as pd

from data_utils import setup_environment
from models.mcnnm import MCNNMEstimator

setup_paths, DataLoader, ForecastEngine = setup_environment()

data_dir, base_dir, tuning_dir, results_dir = setup_paths('scratch')
data_loader = DataLoader(data_dir)
estimator = MCNNMEstimator("mcnnm", results_dir, use_db=True)
engine = ForecastEngine(data_loader, [estimator])

# estimator.X = engine.data_loader.installed_capacity
start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2019-01-31", tz='UTC')
engine.run_forecast(start_date, end_date)
