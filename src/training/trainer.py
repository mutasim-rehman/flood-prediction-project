# src/training/trainer.py
# Contains functions for training the model.

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import sys
from src import config
from src.utils.logger import logger

def train_model():
    """
    Loads the final training data, trains an XGBoost model, evaluates it,
    and saves the model artifact and a feature importance plot.
    """
    logger.info("--- Starting Model Training ---")

    try:
        logger.info(f"Loading processed data from '{config.PROCESSED_FILE_PATH}'...")
        df = pd.read_csv(config.PROCESSED_FILE_PATH)
    except FileNotFoundError:
        logger.error(f"Processed data not found. Please run `main_data_pipeline.py` first.")
        # --- FIX: Exit with a non-zero status code to signal failure ---
        sys.exit(1)

    X = df[config.FEATURE_LIST]
    y = df[config.TARGET_VARIABLE]

    if y.nunique() < 2:
        logger.critical("Training data has only one class. Cannot train a binary classifier.")
        # --- FIX: Exit with a non-zero status code to signal failure ---
        sys.exit(1)

    # Calculate class weight for imbalanced datasets
    scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logger.info(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False # Suppress warning
    )
    model.fit(X_train, y_train)
    logger.success("Model training complete.")

    logger.info("\n--- Model Evaluation on Test Set ---")
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)
    logger.info(f"Classification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{matrix}")

    logger.info(f"Saving trained model to '{config.MODEL_PATH}'...")
    joblib.dump(model, config.MODEL_PATH)
    logger.success("Model saved successfully.")

    # Create and save feature importance plot
    logger.info("Generating feature importance plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(model, ax=ax, title="Feature Importance")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PATH)
    logger.success(f"Feature importance plot saved to '{config.FEATURE_IMPORTANCE_PATH}'")

    logger.success("Model training and artifact saving complete.")

