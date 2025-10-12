# src/data_pipeline/processor.py
# Contains functions for processing raw data and engineering features.

import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import sys
from src import config
from src.utils.logger import logger

def process_and_feature_engineer():
    """
    Loads all raw data sources, merges them, engineers time-series
    and geospatial features, and creates the final labeled training dataset.
    """
    logger.info("--- Starting Data Processing & Feature Engineering ---")

    try:
        logger.info(f"Loading raw hydro-weather data from '{config.RAW_WEATHER_HYDRO_FILEPATH}'...")
        hydro_weather_df = pd.read_csv(config.RAW_WEATHER_HYDRO_FILEPATH, parse_dates=['timestamp'])

        logger.info(f"Loading static terrain data from '{config.TERRAIN_DATA_FILEPATH}'...")
        terrain_df = pd.read_csv(config.TERRAIN_DATA_FILEPATH)

        logger.info(f"Loading ground truth data from '{config.GROUND_TRUTH_PATH}'...")
        floods_df = pd.read_csv(config.GROUND_TRUTH_PATH, parse_dates=['event_date'])
    except FileNotFoundError as e:
        logger.error(f"Cannot find data file: {e}. Please run the data collection step first.")
        # --- FIX: Exit with a non-zero status code to signal failure ---
        sys.exit(1)

    logger.info("Merging terrain data with hydro-weather data...")
    merged_df = pd.merge(hydro_weather_df, terrain_df, on=['lat', 'lon'], how='left')
    merged_df.sort_values(by=['lat', 'lon', 'timestamp'], inplace=True)

    # Forward-fill missing river discharge data
    merged_df['river_discharge_m3s'] = merged_df.groupby(['lat', 'lon'])['river_discharge_m3s'].ffill()

    logger.info("Engineering time-series features (this may take a while)...")
    merged_df['rainfall_24hr_avg'] = merged_df.groupby(['lat', 'lon'])['rainfall_mm_per_hr'].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean())
    merged_df['rainfall_72hr_avg'] = merged_df.groupby(['lat', 'lon'])['rainfall_mm_per_hr'].transform(
        lambda x: x.rolling(window=72, min_periods=1).mean())
    merged_df['month'] = merged_df['timestamp'].dt.month
    merged_df['day_of_year'] = merged_df['timestamp'].dt.dayofyear
    merged_df['hour'] = merged_df['timestamp'].dt.hour
    logger.info("Time-series features created: rolling averages, date components.")

    logger.info("Creating target variable by mapping ground truth events to time-series data...")
    merged_df[config.TARGET_VARIABLE] = 0

    unique_grid_points_df = merged_df[['lat', 'lon']].drop_duplicates()
    grid_coords = np.deg2rad(unique_grid_points_df.values)
    flood_coords = np.deg2rad(floods_df[['lat', 'lon']].values)
    tree = cKDTree(grid_coords)

    search_radius_km = 150  # Search radius in kilometers
    search_radius_rad = search_radius_km / 6371.0  # Earth radius in km
    time_window_days = 14
    nearby_indices = tree.query_ball_point(flood_coords, r=search_radius_rad)

    for i, flood_row in floods_df.iterrows():
        nearby_points_df = unique_grid_points_df.iloc[nearby_indices[i]]
        if nearby_points_df.empty: continue

        start_window = flood_row['event_date'] - pd.Timedelta(days=time_window_days)

        for _, point_row in nearby_points_df.iterrows():
            mask = (
                (merged_df['lat'] == point_row['lat']) &
                (merged_df['lon'] == point_row['lon']) &
                (merged_df['timestamp'] >= start_window) &
                (merged_df['timestamp'] <= flood_row['event_date'])
            )
            merged_df.loc[mask, config.TARGET_VARIABLE] = 1

    labeled_points = merged_df[config.TARGET_VARIABLE].sum()
    logger.info(f"Labeled {labeled_points} data points as flood precursors.")
    if labeled_points == 0:
        logger.critical("No flood events were matched to the time-series data. The model cannot be trained.")
        # --- FIX: Exit with a non-zero status code to signal failure ---
        sys.exit(1)

    final_df = merged_df[config.FEATURE_LIST + [config.TARGET_VARIABLE]].copy()
    final_df.fillna(0, inplace=True)

    logger.info(f"Saving final processed dataset to '{config.PROCESSED_FILE_PATH}'...")
    final_df.to_csv(config.PROCESSED_FILE_PATH, index=False)
    logger.success("Data processing and feature engineering complete.")

