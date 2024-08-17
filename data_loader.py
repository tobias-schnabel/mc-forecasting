import os
import re
from datetime import datetime, timedelta
from typing import Dict
from data_utils import add_calendar_variables
import pandas as pd
from functools import lru_cache
import pytz

class DataLoader:
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.data = {}
        self.installed_capacity = None
        self.installed_capacity_year = None
        self.data_min_date = None
        self.data_max_date = None
        self.utc = pytz.UTC
        # TODO: Add option to set cache size

    def load_data(self):
        installed_capacity_files = []
        for file in os.listdir(self.data_folder):
            if file.endswith('.parquet'):
                key = os.path.splitext(file)[0]
                file_path = os.path.join(self.data_folder, file)

                if key.startswith('installed_capacity_'):
                    installed_capacity_files.append((key, file_path))
                else:
                    self.data[key] = pd.read_parquet(file_path)
                    if not self.data[key].index.tz:
                        self.data[key].index = self.data[key].index.tz_localize('UTC')

                    if self.data_min_date is None or self.data[key].index.min() < self.data_min_date:
                        self.data_min_date = self.data[key].index.min()
                    if self.data_max_date is None or self.data[key].index.max() > self.data_max_date:
                        self.data_max_date = self.data[key].index.max()

        if installed_capacity_files:
            installed_capacity_files.sort(key=lambda x: int(re.findall(r'\d+', x[0])[0]), reverse=True)
            key, file_path = installed_capacity_files[0]
            self.installed_capacity = pd.read_parquet(file_path)
            self.installed_capacity_year = int(re.findall(r'\d+', key)[0])
            print(f"Using installed capacity data from {self.installed_capacity_year}")

        # Add calendar features to coal and gas (time-varying) data
        self.data['coal_gas_cal'] = add_calendar_variables(self.data['coal_gas_data'])

    @lru_cache(maxsize=128)
    def get_slice(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        start_date_utc = self.utc.localize(start_date) if start_date.tzinfo is None else start_date
        end_date_utc = self.utc.localize(end_date) if end_date.tzinfo is None else end_date

        sliced_data = {key: df[(df.index >= start_date_utc) & (df.index < end_date_utc)]
                       for key, df in self.data.items()}
        if self.installed_capacity is not None:
            sliced_data['installed_capacity'] = self.installed_capacity
        return sliced_data

    @lru_cache(maxsize=128)
    def get_next_day(self, date: datetime) -> Dict[str, pd.DataFrame]:
        date_utc = self.utc.localize(date) if date.tzinfo is None else date
        start_date = date_utc.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        end_date = start_date + timedelta(days=1)
        return self.get_slice(start_date, end_date)

    @lru_cache(maxsize=128)
    def get_next_day_with_naive(self, date: datetime) -> Dict[str, pd.DataFrame]:
        date_utc = self.utc.localize(date) if date.tzinfo is None else date
        next_day_data = self.get_next_day(date_utc)
        # TODO: remove actual prices
        previous_day = date_utc - timedelta(days=1)
        naive_forecast = self.get_slice(previous_day, date_utc)['day_ahead_prices']
        next_day_data['naive_forecast'] = naive_forecast
        return next_day_data

    @lru_cache(maxsize=128)
    def get_next_day_features(self, date: datetime) -> Dict[str, pd.DataFrame]:
        next_day_data = self.get_next_day_with_naive(date)
        next_day_data.pop('day_ahead_prices', None)
        return next_day_data

    def get_available_date_range(self) -> tuple:
        if self.data_min_date and self.data_max_date:
            return self.data_min_date, self.data_max_date
        return None

    def clear_cache(self):
        self.get_slice.cache_clear()
        self.get_next_day.cache_clear()
        self.get_next_day_with_naive.cache_clear()
        self.get_next_day_features.cache_clear()