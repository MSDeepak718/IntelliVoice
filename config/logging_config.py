"""
IntelliVoice — Structured Logging Configuration

Uses structlog for structured, colorized, context-rich logging.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application."""

    # Set the root logger level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Suppress noisy third-party loggers
    for logger_name in [
        "urllib3",
        "httpcore",
        "httpx",
        "transformers",
        "accelerate",
        "torch",
        "torchaudio",
        "huggingface_hub",
        "filelock",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=40,
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a named structured logger."""
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    return logger
