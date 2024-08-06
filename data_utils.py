import os
import pandas as pd
from tqdm.auto import tqdm
from entsoe.exceptions import NoMatchingDataError
from retrying import retry
from datetime import datetime, timedelta


@retry(stop_max_attempt_number=5, wait_fixed=5000)
def query_and_save(query_func, filename_template, countries, start_year, end_year, overwrite=False, **kwargs):
    """
    Query data for multiple countries and years, and save the results as parquet files.

    This function queries data using the provided query function for each country and year
    in the specified range. It saves the results as parquet files in a directory structure
    organized by query type and country. The function checks for existing files and only
    re-queries and overwrites data if it's incomplete or if overwrite is set to True.

    For the current year, the function queries data up to two days before the current date.

    Args:
        query_func (callable): The function to use for querying data.
        filename_template (str): Template for the output filename.
        countries (list): List of country codes to query data for.
        start_year (int): The first year to query data for.
        end_year (int): The last year to query data for.
        overwrite (bool, optional): Whether to overwrite existing complete files. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the query function.

    Returns:
        None

    Prints:
        Summary of the operation, including successful queries, countries with no data,
        and countries where errors occurred.
    """
    no_data_countries = []
    error_countries = []
    nan_summary = {}
    successful_queries = 0

    dir_name = filename_template.split('_{}')[0]
    base_dir = os.path.join("raw", dir_name)
    os.makedirs(base_dir, exist_ok=True)

    total_jobs = len(countries) * (end_year - start_year + 1)

    current_date = datetime.now().date()
    two_days_ago = current_date - timedelta(days=2)

    with tqdm(total=total_jobs, desc=f"Processing {query_func.__name__}") as pbar:
        for country in countries:
            country_dir = os.path.join(base_dir, country.lower())
            os.makedirs(country_dir, exist_ok=True)

            for year in range(start_year, end_year + 1):
                start = pd.Timestamp(f"{year}0101", tz="UTC")
                if year == current_date.year:
                    end = pd.Timestamp(two_days_ago, tz="UTC")
                else:
                    end = pd.Timestamp(f"{year}1231", tz="UTC")

                filename = filename_template.format(f"{country}_{year}")
                filepath = os.path.join(country_dir, filename)

                if os.path.exists(filepath) and not overwrite:
                    existing_data = pd.read_parquet(filepath)
                    if existing_data.index[0].normalize() == start and existing_data.index[-1].normalize() == end:
                        pbar.update(1)
                        successful_queries += 1
                        continue

                try:
                    data = query_func(country_code=country, start=start, end=end, **kwargs)

                    df_out = pd.DataFrame(data)

                    if df_out.empty:
                        no_data_countries.append(f"{country}_{year}")
                        pbar.update(1)
                        continue

                    nan_count = df_out.isna().sum().sum()
                    total_count = df_out.size
                    nan_summary[f"{country}_{year}"] = (nan_count, total_count)

                    df_out.to_parquet(filepath, index=True)

                    successful_queries += 1

                except NoMatchingDataError:
                    no_data_countries.append(f"{country}_{year}")
                except Exception as e:
                    tqdm.write(f"Error querying data for {country} in {year}: {e}")
                    error_countries.append(f"{country}_{year}")

                pbar.update(1)

    print("\n Summary:")
    for country_year, (nan_count, total_count) in nan_summary.items():
        if nan_count > 0:
            print(f"{country_year}: {nan_count} NaNs out of {total_count} datapoints")

    print(f"\nSuccessful queries: {successful_queries} / {total_jobs} country-years")

    if no_data_countries:
        print(f"\nNo matching data for: {', '.join(no_data_countries)}")

    if error_countries:
        print(f"\nErrors occurred for: {', '.join(error_countries)}")