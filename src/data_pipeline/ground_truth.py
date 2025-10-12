# src/data_pipeline/ground_truth.py
# Contains functions for creating the ground truth dataset of historical flood events.

import pandas as pd
from src import config
from src.utils.logger import logger

def create_real_ground_truth_file():
    """
    Creates a CSV file of historical flood events based on a manually
    compiled list of real-world data. This represents the "y" variable
    for our supervised learning model.
    """
    logger.info("--- Creating Ground Truth File from Real Data ---")

    if config.GROUND_TRUTH_PATH.exists():
        logger.info(f"Ground truth file already exists at '{config.GROUND_TRUTH_PATH}'. Skipping creation.")
        logger.info("(Delete the file if you want to regenerate it).")
        return

    logger.info("Compiling a list of major historical flood events in Pakistan...")

    # This is a starter list of real, significant flood events.
    # For a robust model, this list should be expanded with more research.
    # Severity is on a scale of 1 (localized) to 3 (catastrophic).
    flood_events = {
        'event_date': [
            '2010-07-28',  # The Great 2010 Superfloods (start date in KPK)
            '2011-08-11',  # Sindh floods
            '2014-09-07',  # Kashmir and Punjab floods
            '2017-07-30',  # Karachi urban flooding
            '2020-08-25',  # More severe Karachi urban flooding
            '2022-08-15',  # The devastating 2022 floods (a representative mid-point)
            '2023-04-29'   # Spring floods in KPK and Punjab
        ],
        'lat': [34.01, 25.39, 31.52, 24.86, 24.90, 28.37, 32.93],
        'lon': [71.52, 68.85, 74.35, 67.01, 67.05, 68.45, 72.36],
        'severity': [3, 2, 2, 1, 2, 3, 1]
    }

    floods_df = pd.DataFrame(flood_events)
    floods_df['event_date'] = pd.to_datetime(floods_df['event_date'])

    floods_df.to_csv(config.GROUND_TRUTH_PATH, index=False)
    logger.success(f"Real flood data saved to '{config.GROUND_TRUTH_PATH}'.")
    logger.info(f"Contains {len(floods_df)} major flood events for training.")
