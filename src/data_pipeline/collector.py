# src/data_pipeline/collector.py
# Contains functions for collecting raw data from external APIs.

import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
import time
from src import config
from src.utils.logger import logger


def setup_directories():
    """Creates necessary directories defined in the config if they don't exist."""
    logger.info("Setting up project directories...")
    config.RAW_API_DIR.mkdir(parents=True, exist_ok=True)
    config.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory check complete. Raw data will be saved in '{config.RAW_API_DIR}'.")


def collect_static_terrain_data(coordinates):
    """
    Collects static elevation and slope data for given coordinates.
    This is a one-time operation per location.
    """
    if config.TERRAIN_DATA_FILEPATH.exists():
        logger.info(f"Terrain data already exists at '{config.TERRAIN_DATA_FILEPATH}'. Skipping.")
        return

    logger.info("--- Collecting Static Terrain Data (One-Time Operation) ---")
    url = "https://api.open-meteo.com/v1/elevation"
    terrain_data = []

    for lat, lon in coordinates:
        params = {"latitude": lat, "longitude": lon}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            elevation = response.json()['elevation'][0]

            params_north = {"latitude": lat + 0.01, "longitude": lon}
            params_east = {"latitude": lat, "longitude": lon + 0.01}
            ele_north = requests.get(url, params=params_north, timeout=10).json()['elevation'][0]
            ele_east = requests.get(url, params=params_east, timeout=10).json()['elevation'][0]
            slope = ((ele_north - elevation) ** 2 + (ele_east - elevation) ** 2) ** 0.5

            terrain_data.append({'lat': lat, 'lon': lon, 'elevation_m': elevation, 'slope_degrees': slope})
            logger.info(f"Fetched terrain for ({lat}, {lon})")
            time.sleep(0.5)  # Be respectful to the API
        except Exception as e:
            logger.error(f"Could not fetch terrain for ({lat}, {lon}): {e}")

    if terrain_data:
        df_terrain = pd.DataFrame(terrain_data)
        df_terrain.to_csv(config.TERRAIN_DATA_FILEPATH, index=False)
        logger.success(f"Terrain data saved to '{config.TERRAIN_DATA_FILEPATH}'.")


def intelligent_hydro_weather_collector(coordinates, high_alt_coord):
    """
    Maintains a local CSV of historical weather data. Fetches new data if the
    local file is outdated.
    """
    logger.info("--- Starting Intelligent Hydro-Weather Data Collector ---")
    output_filename = config.RAW_WEATHER_HYDRO_FILEPATH
    start_date_str = "2010-01-01"
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    is_update = False

    if output_filename.exists():
        logger.info(f"Local data file found: '{output_filename}'")
        try:
            df = pd.read_csv(output_filename, parse_dates=['timestamp'])
            last_date_in_file = df['timestamp'].max()
            start_date_for_update = last_date_in_file + timedelta(days=1)

            if start_date_for_update.date() >= datetime.now().date():
                logger.success("Hydro-weather data is already up-to-date. No download needed.")
                return
            start_date_str = start_date_for_update.strftime('%Y-%m-%d')
            is_update = True
            logger.info(f"Updating data from {start_date_str} to {end_date_str}...")
        except Exception as e:
            logger.warning(f"Could not read existing file. Performing a full download. Error: {e}")
            output_filename.unlink()
    else:
        logger.info("No local data file found. Performing full download from 2010 to present.")

    # Fetch high-altitude temperature first (glacial melt proxy)
    logger.info(f"Fetching high-altitude temperature proxy from ({high_alt_coord[0]}, {high_alt_coord[1]})...")
    proxy_params = {
        "latitude": high_alt_coord[0], "longitude": high_alt_coord[1],
        "start_date": start_date_str, "end_date": end_date_str,
        "hourly": "temperature_2m", "timezone": "auto"
    }
    proxy_response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=proxy_params, timeout=30)
    proxy_data = proxy_response.json()
    df_proxy = pd.DataFrame(proxy_data['hourly'])
    df_proxy.rename(columns={'time': 'timestamp', 'temperature_2m': 'high_alt_temp_proxy'}, inplace=True)
    df_proxy['timestamp'] = pd.to_datetime(df_proxy['timestamp'])

    # Fetch main weather and river data
    new_data_list = []
    url = "https://archive-api.open-meteo.com/v1/archive"
    for i, (lat, lon) in enumerate(coordinates):
        # --- FIX: Request only precipitation, as river_discharge is not available for all general coordinates ---
        # and causes a 400 Bad Request error. We will add a placeholder column for it later.
        params = {
            "latitude": lat, "longitude": lon, "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "precipitation", "timezone": "auto"
        }
        try:
            logger.info(f"Fetching data for location {i + 1}/{len(coordinates)} ({lat}, {lon})...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'hourly' not in data or not data['hourly']['time']:
                logger.warning(f"No data returned for ({lat}, {lon}). Skipping.")
                continue

            df_temp = pd.DataFrame(data['hourly'])
            df_temp['lat'] = data['latitude']
            df_temp['lon'] = data['longitude']

            # Add a placeholder for river discharge, as it's not requested from the API.
            # The data processor will forward-fill this data where it's missing.
            df_temp['river_discharge_m3s'] = np.nan

            new_data_list.append(df_temp)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Skipping location ({lat}, {lon}) due to error: {e}")

    if not new_data_list:
        logger.critical("No new data was fetched. Collector is stopping. The pipeline cannot continue.")
        # Return False to indicate failure
        return False

    logger.info("Processing and saving new data...")
    new_df = pd.concat(new_data_list, ignore_index=True)
    new_df.rename(columns={
        'time': 'timestamp',
        'precipitation': 'rainfall_mm_per_hr',
    }, inplace=True)
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    final_df = pd.merge(new_df, df_proxy, on='timestamp', how='left')

    if is_update:
        final_df.to_csv(output_filename, mode='a', header=False, index=False)
        logger.success(f"Successfully appended new data to '{output_filename}'.")
    else:
        final_df.to_csv(output_filename, index=False)
        logger.success(f"Successfully created new data file at '{output_filename}'.")
    # Return True to indicate success
    return True

