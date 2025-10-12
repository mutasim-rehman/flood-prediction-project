# src/prediction/predictor.py
# Contains functions for loading the trained model and making predictions.

import joblib
import pandas as pd
from src import config
from src.utils.logger import logger

def load_model():
    """Loads the trained XGBoost model from the file."""
    try:
        logger.info(f"Loading model from '{config.MODEL_PATH}'...")
        model = joblib.load(config.MODEL_PATH)
        logger.success("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at '{config.MODEL_PATH}'.")
        logger.error("Please run `main_train.py` first to train and save the model.")
        return None

def predict_flood_risk(input_df: pd.DataFrame):
    """
    Makes flood risk predictions on new data using the trained model.

    Args:
        input_df (pd.DataFrame): A DataFrame containing the input features.
                                 Its columns must match `config.FEATURE_LIST`.

    Returns:
        A tuple containing:
        - list of predictions ("Flood Risk" or "No Flood Risk")
        - list of probabilities for the positive class (flood)
        Returns (None, None) if the model cannot be loaded.
    """
    model = load_model()
    if model is None:
        return None, None

    # Ensure columns are in the correct order
    input_df_ordered = input_df[config.FEATURE_LIST]

    logger.info(f"Making predictions on {len(input_df_ordered)} data points...")
    probabilities = model.predict_proba(input_df_ordered)[:, 1]

    predictions = ["Flood Risk" if prob >= config.PREDICTION_THRESHOLD else "No Flood Risk" for prob in probabilities]
    logger.success("Prediction complete.")

    return predictions, probabilities
