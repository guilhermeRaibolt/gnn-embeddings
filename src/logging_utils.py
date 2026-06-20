"""Logging configuration: write every message to both stdout and a timestamped file."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

LOGGER_NAME = "gnn_experiment"


def setup_logging(logs_dir: Path, level: int = logging.INFO) -> Tuple[logging.Logger, Path]:
    """Create ``logs/`` (if needed) and attach a file + console handler.

    Returns the configured logger and the path of the log file, which is named
    ``logs/gnn_experiment_[TIMESTAMP].log`` as required by the experiment spec.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"gnn_experiment_{timestamp}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()  # idempotent: avoid duplicate handlers on re-entry

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def get_logger() -> logging.Logger:
    """Return the shared experiment logger (after :func:`setup_logging` ran)."""
    return logging.getLogger(LOGGER_NAME)
