"""Hoax detection module initialization."""
from pathlib import Path
import os

__version__ = "1.0.0"

# Config paths
CONFIG_DIR = Path(__file__).parent / "config"
TRUSTED_DOMAINS_PATH = CONFIG_DIR / "trusted_domains.json"
