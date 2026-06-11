"""
utils.py
--------
General-purpose helper functions shared across the project.
"""

import os
import logging


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Return a configured root logger."""
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    return logging.getLogger()


def save_figure(fig, path: str, dpi: int = 150) -> None:
    """Save a matplotlib figure to *path*, creating parent dirs as needed."""
    ensure_dirs(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dictionary."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
