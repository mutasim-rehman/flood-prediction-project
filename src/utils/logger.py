# src/utils/logger.py
# Centralized logger configuration for the entire project.

import sys
from loguru import logger
from src.config import LOG_DIR

# Ensure the log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove default handler to avoid duplicate console outputs
logger.remove()

# Configure a handler for console output with a specific format and color
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Configure a handler to write logs to a file
# This will create a new log file each time the application runs
log_file_path = LOG_DIR / "app.log"
logger.add(
    log_file_path,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",  # Rotates the file when it reaches 10 MB
    retention="7 days", # Keeps logs for 7 days
    enqueue=True,      # Make logging non-blocking
    backtrace=True,
    diagnose=True
)

logger.info("Logger configured. All outputs will be logged.")
