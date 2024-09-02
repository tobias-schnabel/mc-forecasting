import pandas as pd

from data_utils import setup_environment
from models.elastic_net_regression import ElasticNetEstimator
from models.mcnnm import MCNNMEstimator

setup_paths, DataLoader, ForecastEngine = setup_environment()

data_dir, base_dir, tuning_dir, results_dir = setup_paths()
data_loader = DataLoader(data_dir)

mc = MCNNMEstimator("MCNNM", results_dir, use_db=True)
# lasso = LASSOEstimator("LASSO_panel", results_dir, use_db=True)
# lear = LEAREstimator("LEAR_MV", results_dir, use_db=True)
en = ElasticNetEstimator("ElasticNet", results_dir, use_db=True)

engine = ForecastEngine(data_loader, [mc, en])

mc.X = engine.data_loader.installed_capacity
start_date = pd.Timestamp("2019-01-01", tz='UTC')
end_date = pd.Timestamp("2020-12-31", tz='UTC')
engine.run_forecast(start_date, end_date)
