"""Utility functions for the emotive currents package."""

import logging
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def validate_input(data: Any) -> bool:
    """Validate input data."""
    if data is None:
        return False
    return not (isinstance(data, str) and len(data.strip()) == 0)


def process_batch(items: list[str]) -> list[dict]:
    """Process a batch of items."""
    results = []
    for item in items:
        if validate_input(item):
            results.append({"item": item, "processed": True})
        else:
            results.append({"item": item, "processed": False, "error": "Invalid input"})
    return results
