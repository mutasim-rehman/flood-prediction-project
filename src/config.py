# src/config.py
# Central configuration file for the project.
# Contains all paths, constants, and parameters to ensure consistency
# and make the pipeline easy to modify.

from pathlib import Path

# --- Core Project Directories ---
# Using Path for OS-agnostic path handling.
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
RAW_API_DIR = DATA_DIR / "raw_api"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# --- File Paths ---
# Using static filenames makes the pipeline much more robust.
RAW_WEATHER_HYDRO_FILEPATH = RAW_API_DIR / "historical_weather_hydro_data.csv"
TERRAIN_DATA_FILEPATH = RAW_API_DIR / "static_terrain_data.csv"
GROUND_TRUTH_PATH = GROUND_TRUTH_DIR / "historical_floods.csv"
PROCESSED_FILE_PATH = PROCESSED_DATA_DIR / "final_training_dataset.csv"

# --- Data Collection Parameters ---
# Coordinates for major cities/flood-prone areas in Pakistan
MAIN_COORDINATES = [
    (24.86, 67.01),  # Karachi
    (31.52, 74.35),  # Lahore
    (33.68, 73.04),  # Islamabad
    (30.17, 66.99),  # Quetta
    (34.01, 71.52),  # Peshawar
    (25.39, 68.35),  # Hyderabad
    (30.15, 71.48)   # Multan
]
# A single high-altitude coordinate in the north for the glacial melt proxy
GLACIAL_PROXY_COORDINATE = (35.92, 74.30)  # Near Gilgit

# --- Feature Engineering & Model Training ---
# Expanded feature list including new hydrological and topographical data
FEATURE_LIST = [
    'lat', 'lon', 'rainfall_mm_per_hr', 'rainfall_24hr_avg',
    'rainfall_72hr_avg', 'month', 'day_of_year', 'hour',
    'elevation_m', 'slope_degrees',         # Terrain features
    'river_discharge_m3s',                 # River discharge feature
    'high_alt_temp_proxy'                  # Glacial melt proxy feature
]
TARGET_VARIABLE = 'flood_event'

# --- Model Artifacts ---
MODEL_PATH = MODEL_DIR / "flood_prediction_xgboost_model.joblib"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.png"

# --- Prediction ---
PREDICTION_THRESHOLD = 0.5
