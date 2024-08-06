import os
import pandas as pd
from tqdm.auto import tqdm
from entsoe.exceptions import NoMatchingDataError
from retrying import retry
from datetime import datetime, timedelta
from glob import glob
from typing import List, Tuple

# Define the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data_path(sub_path: str) -> str:
    """
    Get the absolute path for a data file or directory.

    Args:
    sub_path (str): Relative path from the data directory

    Returns:
    str: Absolute path
    """
    return os.path.join(PROJECT_ROOT, 'data', sub_path)


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
    base_dir = get_data_path(os.path.join("raw", dir_name))
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
                    # Include all hours of December 31st
                    end = pd.Timestamp(f"{year + 1}0101", tz="UTC") - pd.Timedelta(seconds=1)


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


def load_data(variable: str, years: List[int], countries: List[str]) -> List[pd.DataFrame]:
    base_path = get_data_path('raw')
    dfs = []

    for country in countries:
        for year in years:
            path = os.path.join(base_path, variable, country.lower(), f"{variable}_{country}_{year}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df['country'] = country
                dfs.append(df)

    return dfs


def check_time_continuity(dfs: List[pd.DataFrame]) -> Tuple[bool, List[pd.DataFrame]]:
    """
    Check if the dataframes have continuous time indices without gaps.

    Args:
    dfs (List[pd.DataFrame]): List of dataframes to check

    Returns:
    Tuple[bool, List[pd.DataFrame]]:
        - Boolean indicating if all dataframes are continuous
        - List of dataframes with gaps (empty if all are continuous)
    """
    problematic_dfs = []

    for df in dfs:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        if not df.index.equals(expected_index):
            problematic_dfs.append(df)

    return len(problematic_dfs) == 0, problematic_dfs


def check_date_gaps(df: pd.DataFrame) -> None:
    """
    Check for gaps in the date index of a dataframe.

    Args:
    df (pd.DataFrame): Dataframe to check, with a DatetimeIndex

    Raises:
    ValueError: If gaps are found in the date index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Check if the index is sorted
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Calculate the time difference between consecutive entries
    time_diff = df.index.to_series().diff()

    # Find gaps (assuming hourly data)
    gaps = time_diff[time_diff > pd.Timedelta(hours=1)]

    if not gaps.empty:
        gap_info = "\n".join([f"{gap.index}: {gap}" for gap in gaps.items()])
        raise ValueError(f"Gaps found in the date index:\n{gap_info}")
    else:
        print("No gaps found in the date index.")


def load_installed_capacity(year: int) -> pd.DataFrame:
    base_path = get_data_path('raw/installed_generation_capacity')
    dfs = []

    for country_folder in os.listdir(base_path):
        path = os.path.join(base_path, country_folder, f"installed_generation_capacity_{country_folder}_{year}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['country'] = country_folder.upper()
            dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    # make country the index
    result.set_index('country', inplace=True)
    return result


def drop_non_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all values that are not at full hours.

    Args:
    df (pd.DataFrame): Input dataframe with datetime index

    Returns:
    pd.DataFrame: Dataframe with only full hour data points
    """
    return df[df.index.minute == 0]


def total_columns(df: pd.DataFrame) -> pd.Series:
    """
    Total all numeric columns for each row in a DataFrame, ignoring NaN values.
    :param df:
    :return:
    """
    # only total numeric columns
    df = df.select_dtypes(include='number')
    return df.sum(axis=1, skipna=True)


def load_coal_gas_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load coal and gas data for a specified date range, padding non-trading days and resampling to desired frequency.
    Includes diagnostic information.

    Args:
    start_date (str): Start date in 'YYYY-MM-DD'
    end_date (str): End date in 'YYYY-MM-DD'

    Returns:
    pd.DataFrame: Dataframe containing coal and gas data for the specified date range in hourly frequency
    """

    def load_cg_data(file_pattern):
        files = glob(get_data_path(file_pattern))
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            dfs.append(df)
        return pd.concat(dfs).sort_index()

    coal_df = load_cg_data('raw/coal/coal_*.parquet')
    coal_df['Date'] = pd.to_datetime(coal_df['Date']).dt.normalize()
    coal_df.set_index('Date', inplace=True)

    gas_df = load_cg_data('raw/gas/gas_*.parquet')
    gas_df['Date'] = pd.to_datetime(gas_df['Date']).dt.normalize()
    gas_df.set_index('Date', inplace=True)

    # Merge coal and gas data
    daily_df = pd.concat([coal_df, gas_df], axis=1, join='outer')
    daily_df.sort_index(inplace=True)
    daily_df = daily_df[start_date:end_date]

    # Create a new DataFrame with hourly frequency
    hourly_index = pd.date_range(start=start_date, end=end_date, freq='h')
    hourly_df = pd.DataFrame(index=hourly_index, columns=daily_df.columns)

    # Fill the hourly DataFrame with daily values
    for date, row in daily_df.iterrows():
        hourly_df.loc[date.strftime('%Y-%m-%d'), :] = row.values

    # Forward fill the hourly values
    hourly_df = hourly_df.ffill()
    hourly_df = hourly_df.infer_objects(copy=False)

    # Handle the exception for 2019-01-01
    if '2019-01-02' in daily_df.index:
        jan_2_values = hourly_df.loc['2019-01-02 00:00:00'].values
        hourly_df.loc['2019-01-01'] = jan_2_values

    return hourly_df


def tall_to_wide(df, index_cols, value_col):
    """
    Convert a tall dataframe to a wide dataframe.
    :param df: Input dataframe
    :param index_cols: Columns to use as index
    :param value_col: Column to use as values
    :return: Wide dataframe
    """
    return df.pivot(index=index_cols, columns='datetime', values=value_col)


def load_day_ahead_prices(start_date: str, end_date: str, countries: List[str]) -> pd.DataFrame:
    """
    Load day-ahead prices for specified countries within the given date range into a matrix.

    Args:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    countries (List[str]): List of country codes to load data for

    Returns:
    pd.DataFrame: Matrix of day-ahead prices with dates as rows and countries as columns
    """
    # Convert start and end dates to datetime in UTC
    start = pd.to_datetime(start_date).tz_localize('UTC')
    end = pd.to_datetime(end_date).tz_localize('UTC')

    # Determine the years to load based on the date range
    years = list(range(start.year, end.year + 1))

    # Remove duplicates from the countries list
    countries = list(dict.fromkeys(countries))

    # Load the data
    dfs = load_data('day_ahead_prices', years, countries)

    # Check if any data was loaded
    if not dfs:
        raise ValueError(f"No data available for the specified countries and date range.")

    # Process each country separately
    country_dfs = {}
    for country in countries:
        country_data = [df for df in dfs if df['country'].iloc[0] == country]
        if country_data:
            # Concatenate all years for this country
            country_df = pd.concat(country_data)
            # Ensure the index is timezone-aware UTC
            if country_df.index.tz is None:
                country_df.index = country_df.index.tz_localize('UTC')
            else:
                country_df.index = country_df.index.tz_convert('UTC')

            # Count duplicates before removal
            duplicate_count = country_df.index.duplicated().sum()

            # Sort index and remove duplicates, keeping the last occurrence
            country_df = country_df.sort_index().groupby(level=0).last()

            # Print information about dropped duplicates
            if duplicate_count > 0:
                print(f"Dropped {duplicate_count} duplicate entries for {country}")

            # Select only the price column and rename it to the country code
            country_dfs[country] = country_df.iloc[:, 0].rename(country)

    # Combine all country dataframes
    result_df = pd.concat(country_dfs.values(), axis=1)

    # Create a complete datetime index for the entire range
    full_index = pd.date_range(start=start, end=end, freq='h', tz='UTC')

    # Reindex the result dataframe to fill any gaps
    result_df = result_df.reindex(full_index)

    # Detailed analysis of missing data
    print("\nDetailed missing data analysis:")
    for country in countries:
        missing_mask = result_df[country].isnull()
        missing_count = missing_mask.sum()
        if missing_count > 0:
            print(f"\n{country}:")
            print(f"  Total missing entries: {missing_count}")

            # Find contiguous ranges of missing data
            missing_ranges = []
            missing_start = None
            for date, is_missing in missing_mask.items():
                if is_missing and missing_start is None:
                    missing_start = date
                elif not is_missing and missing_start is not None:
                    missing_ranges.append((missing_start, date - pd.Timedelta(hours=1)))
                    missing_start = None
            if missing_start is not None:
                missing_ranges.append((missing_start, missing_mask.index[-1]))

            # Print missing ranges
            for start, end in missing_ranges:
                print(f"  Missing range: {start} to {end}")

            # Check for specific patterns
            missing_by_year = missing_mask.groupby(missing_mask.index.year).sum()
            print("  Missing entries by year:")
            for year, count in missing_by_year.items():
                if count > 0:
                    print(f"    {year}: {count}")

    return result_df

# new to be used
# def load_day_ahead_prices(start_date: str, end_date: str, countries: List[str]) -> pd.DataFrame:
#     """
#     Load day-ahead prices for specified countries within the given date range into a matrix.
#
#     Args:
#     start_date (str): Start date in 'YYYY-MM-DD' format
#     end_date (str): End date in 'YYYY-MM-DD' format
#     countries (List[str]): List of country codes to load data for
#
#     Returns:
#     pd.DataFrame: Matrix of day-ahead prices with dates as rows and countries as columns
#     """
#     # Convert start and end dates to datetime in UTC
#     start = pd.to_datetime(start_date).tz_localize('UTC')
#     end = pd.to_datetime(end_date).tz_localize('UTC')
#
#     # Determine the years to load based on the date range
#     years = list(range(start.year, end.year + 1))
#
#     # Remove duplicates from the countries list
#     countries = list(dict.fromkeys(countries))
#
#     # Load the data
#     dfs = load_data('day_ahead_prices', years, countries)
#
#     # Check if any data was loaded
#     if not dfs:
#         raise ValueError(f"No data available for the specified countries and date range.")
#
#     # Process each country separately
#     country_dfs = {}
#     for country in countries:
#         country_data = [df for df in dfs if df['country'].iloc[0] == country]
#         if country_data:
#             # Concatenate all years for this country
#             country_df = pd.concat(country_data)
#             # Ensure the index is timezone-aware UTC
#             if country_df.index.tz is None:
#                 country_df.index = country_df.index.tz_localize('UTC')
#             else:
#                 country_df.index = country_df.index.tz_convert('UTC')
#
#             # Sort index and remove duplicates, keeping the last occurrence
#             country_df = country_df.sort_index().groupby(level=0).last()
#
#             # Select only the price column and rename it to the country code
#             country_dfs[country] = country_df.iloc[:, 0].rename(country)
#
#     # Combine all country dataframes
#     result_df = pd.concat(country_dfs.values(), axis=1)
#
#     # Select the date range
#     result_df = result_df.loc[start:end]
#
#     return result_df

def check_date_continuity(df: pd.DataFrame) -> None:
    """
    Check for continuity in the date index of a dataframe.

    Args:
    df (pd.DataFrame): Dataframe to check, with a DatetimeIndex

    Raises:
    ValueError: If discontinuities are found in the date index
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    # Calculate the time difference between consecutive entries
    time_diff = df.index.to_series().diff()

    # Find discontinuities (assuming hourly data)
    discontinuities = time_diff[time_diff != pd.Timedelta(hours=1)]

    if not discontinuities.empty:
        disc_info = "\n".join([f"{disc.index}: {disc}" for disc in discontinuities.items()])
        raise ValueError(f"Discontinuities found in the date index:\n{disc_info}")
    else:
        print("Date index is continuous.")

