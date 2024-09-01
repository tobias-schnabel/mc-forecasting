import pandas as pd

from data_utils import setup_environment

setup_paths, DataLoader, ForecastEngine = setup_environment()
from models.elastic_net_regression import ElasticNetEstimator
data_dir, base_dir, tuning_dir, results_dir = setup_paths('scratch')
data_loader = DataLoader(data_dir)
estimator = ElasticNetEstimator("ElasticNet", results_dir, use_db=True)
engine = ForecastEngine(data_loader, [estimator])
# engine.min_train_window =7
start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2019-12-31", tz='UTC')
engine.run_forecast(start_date, end_date)
