# main_train.py
# =============
# This is the main entry point for Step 2: Model Training.
# It loads the processed dataset and runs the model training and
# evaluation script from the training module.
# ==========================================================

from src.utils.logger import logger
from src.training import trainer

def run_training():
    """
    Executes the model training and evaluation process.
    """
    logger.info("========== STARTING: STEP 2 - MODEL TRAINING ==========")
    print("MAIN_TRAIN: Executing model training pipeline. Check 'logs/app.log' for details.")
    trainer.train_model()
    logger.success("========== COMPLETED: STEP 2 - MODEL TRAINING ==========")
    print("MAIN_TRAIN: Successfully trained and saved the model.")


if __name__ == "__main__":
    run_training()
