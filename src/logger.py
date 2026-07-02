"""
Centralized logging configuration for the Local RAG Agent.

Provides a factory function to create consistently configured loggers
across all modules. Uses the standard library `logging` module with
structured formatting and configurable levels via environment variables.
"""

import logging
import sys
from typing import Optional

from src.config import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create or retrieve a logger with standardized configuration.

    Args:
        name: Logger name, typically `__name__` of the calling module.
        level: Override the default log level for this logger instance.

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
        2024-06-18 01:25:00,000 | INFO     | src.my_module | Application started
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    # Set level
    effective_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))

    # Console handler with colored output
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger
