# main_predict.py
# ===============
# This is the main entry point for Step 3: Prediction.
# It loads the trained model and demonstrates how to make predictions
# on new, sample data points.
# ============================================================

import pandas as pd
from src.utils.logger import logger
from src.prediction import predictor
from src.config import FEATURE_LIST

def run_prediction():
    """
    Demonstrates making predictions with the trained model on sample data.
    """
    logger.info("========== STARTING: STEP 3 - PREDICTION DEMO ==========")
    print("MAIN_PREDICT: Running prediction demo. Check 'logs/app.log' for details.")

    # Example 1: High-risk scenario
    # Heavy monsoon rain in a low-elevation area with already high river discharge.
    high_risk_data = {
        'lat': 25.39, 'lon': 68.35,              # Hyderabad, low elevation
        'rainfall_mm_per_hr': 18.0,
        'rainfall_24hr_avg': 10.2,
        'rainfall_72hr_avg': 5.1,
        'month': 8, 'day_of_year': 238, 'hour': 16,
        'elevation_m': 13, 'slope_degrees': 2.5, # Low elevation, slight slope
        'river_discharge_m3s': 5500,             # High river flow
        'high_alt_temp_proxy': 12.5              # Warm, indicating glacial melt
    }

    # Example 2: Low-risk scenario
    # Light rain in a higher elevation area with low river levels.
    low_risk_data = {
        'lat': 33.68, 'lon': 73.04,              # Islamabad, higher elevation
        'rainfall_mm_per_hr': 0.2,
        'rainfall_24hr_avg': 0.1,
        'rainfall_72hr_avg': 0.0,
        'month': 11, 'day_of_year': 320, 'hour': 10,
        'elevation_m': 540, 'slope_degrees': 15.0, # Higher elevation and slope
        'river_discharge_m3s': 450,               # Low river flow
        'high_alt_temp_proxy': -5.0               # Cold, no glacial melt
    }

    # Create a DataFrame for batch prediction
    prediction_df = pd.DataFrame([high_risk_data, low_risk_data], columns=FEATURE_LIST)

    # Make predictions
    predictions, probabilities = predictor.predict_flood_risk(prediction_df)

    if predictions is not None:
        print("\n--- PREDICTION RESULTS ---")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Data Point #{i+1}:")
            print(f"  Prediction: {pred}")
            print(f"  Probability of Flood: {prob:.2%}\n")
            logger.info(f"Prediction for input {i+1}: {pred} (Probability: {prob:.2%})")

    logger.success("========== COMPLETED: STEP 3 - PREDICTION DEMO ==========")
    print("MAIN_PREDICT: Prediction demo finished successfully.")


if __name__ == "__main__":
    run_prediction()
