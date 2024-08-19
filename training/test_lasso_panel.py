import pandas as pd

from data_utils import setup_environment

setup_paths, DataLoader, ForecastEngine = setup_environment()
from models.lasso_panel import LEAREstimator
data_dir, base_dir, tuning_dir, results_dir = setup_paths('scratch')
data_loader = DataLoader(data_dir)
estimator = LEAREstimator("LEAR_MV", results_dir, use_db=True)
engine = ForecastEngine(data_loader, [estimator])
engine.min_train_window = 8
start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2019-03-31", tz='UTC')
engine.run_forecast(start_date, end_date)
