"""
Shared logging configuration for fall detection scripts.
Writes to /home/{user}/logs/ with daily rotation.
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Log directory and file
LOG_DIR = os.path.expanduser("~/logs")
LOG_FILE = os.path.join(LOG_DIR, "fall-detection.log")


def setup_logging(script_name: str) -> logging.Logger:
    """
    Set up logging with daily rotation.

    Args:
        script_name: Name of the script (e.g., 'fall-detector', 'adl-collector')

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger("fall-detection")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # File handler with daily rotation
    # Keeps 30 days of logs, rotates at midnight
    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.INFO)

    # Console handler (also captured by systemd)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format: timestamp [script] level - message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(script_name)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add script name to all log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.script_name = script_name
        return record

    logging.setLogRecordFactory(record_factory)

    logger.info(f"Logging initialized for {script_name}")
    logger.info(f"Log file: {LOG_FILE}")

    return logger
