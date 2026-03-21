"""
logger.py — Centralized logging setup.

Why: print() statements disappear and can't be reviewed later.
A proper logger writes to both the console AND a log file, with
timestamps and severity levels (INFO, WARNING, ERROR).
"""

import logging
import os
from config import config


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for any module that calls it.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("Starting ingestion...")
        logger.error("Failed to load PDF: %s", path)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Format ────────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (INFO and above) ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # ── File handler (DEBUG and above — captures everything) ──────────────────
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger