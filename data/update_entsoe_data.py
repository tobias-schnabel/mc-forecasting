from entsoe import EntsoePandasClient
import os
import sys
import subprocess
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import query_and_save

load_dotenv()  # load env file
entsoe_key = os.getenv("ENTSOE_API_KEY")  # Get API keys from .env file

# Instantiate the ENTSOE Client
client = EntsoePandasClient(api_key=entsoe_key)

# Define the overall time range
start_year = 2019
end_year = 2024

# Define the countries, CORE CCR countries are: AT BE HR CZ FR DE HU LU NL PL RO SK SI
countries = ["AT", "BE", "HR", "CZ", "FR", "DE_LU", "HU", "NL", "PL", "RO", "SK", "SI", "PL", "CH"]

query_and_save(
    query_func=client.query_day_ahead_prices,
    filename_template="day_ahead_prices_{}.parquet",
    countries=countries, start_year=start_year, end_year=end_year, overwrite=True)

query_and_save(
    query_func=client.query_load_forecast,
    filename_template="load_forecast_{}.parquet",
    countries=countries, start_year=start_year, end_year=end_year, overwrite=True)

query_and_save(
    query_func=client.query_generation_forecast,
    filename_template="generation_forecast_{}.parquet",
    countries=countries, start_year=start_year, end_year=end_year, overwrite=True)

query_and_save(
    query_func=client.query_wind_and_solar_forecast,
    filename_template="wind_and_solar_forecast_{}.parquet",
    countries=countries, start_year=start_year, end_year=end_year, overwrite=True)

query_and_save(
    query_func=client.query_installed_generation_capacity,
    filename_template="installed_generation_capacity_{}.parquet",
    countries=countries, start_year=start_year, end_year=end_year, overwrite=True)


def git_add_commit(directory, commit_message):
    try:
        # Stage all changed files in the specified directory
        subprocess.run(['git', 'add', directory], check=True)

        # Commit the changes with the provided commit message
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)

        print("Changes have been staged and committed.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

git_add_commit("data", "Update ENTSOE data")