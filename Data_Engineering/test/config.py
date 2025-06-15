"""Test configuration and constants."""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Base paths
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"
TEST_LOGS_DIR = TEST_DIR / "logs"
TEST_CASES_DIR = TEST_DIR / "cases"
TEST_RESULTS_DIR = TEST_DIR / "results"

# Ensure directories exist
for dir_path in [TEST_DATA_DIR, TEST_LOGS_DIR, TEST_CASES_DIR, TEST_RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Test data constants
ENCODINGS = ['utf-8', 'iso-8859-1', 'windows-1252']
MIN_SEGMENT_LINES = 2
MAX_SEGMENT_LINES = 10

# Configure logging
LOG_FILE = TEST_LOGS_DIR / "test_execution.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)