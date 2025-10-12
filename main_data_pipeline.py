# main_data_pipeline.py
# =====================
# This is the main entry point for Step 1: Data Collection & Processing.
# It orchestrates the execution of all scripts in the data_pipeline module
# to create the final, model-ready dataset.
# ========================================================================

from src.utils.logger import logger
from src.data_pipeline import collector, ground_truth, processor
from src.config import MAIN_COORDINATES, GLACIAL_PROXY_COORDINATE

def run_pipeline():
    """
    Executes the full data collection, ground truth creation, and
    processing pipeline in the correct order.
    """
    logger.info("========== STARTING: STEP 1 - DATA PIPELINE ==========")
    print("MAIN_DATA_PIPELINE: Executing full data collection and processing pipeline. Check 'logs/app.log' for details.")

    # Part 1: Collect raw data from APIs
    logger.info("--- Running Data Collector ---")
    collector.setup_directories()
    collector.collect_static_terrain_data(MAIN_COORDINATES)
    collector.intelligent_hydro_weather_collector(MAIN_COORDINATES, GLACIAL_PROXY_COORDINATE)
    logger.success("--- Data Collector Finished ---")

    # Part 2: Create the historical floods file (ground truth)
    logger.info("--- Running Ground Truth Creator ---")
    ground_truth.create_real_ground_truth_file()
    logger.success("--- Ground Truth Creator Finished ---")

    # Part 3: Process and merge all data sources to create the final dataset
    logger.info("--- Running Data Processor & Feature Engineer ---")
    processor.process_and_feature_engineer()
    logger.success("--- Data Processor Finished ---")

    logger.success("========== COMPLETED: STEP 1 - DATA PIPELINE ==========")
    print("MAIN_DATA_PIPELINE: Successfully created the final training dataset.")


if __name__ == "__main__":
    run_pipeline()
