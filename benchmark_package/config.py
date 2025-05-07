import os
import socket

# System settings
SOURCE = socket.gethostname()
CURRENT_DIR = os.getcwd()
CSV_PATH = os.path.join(CURRENT_DIR, "fake_data.csv")
DB_PATH = os.path.join(CURRENT_DIR, "benchmark_results.db")

# BigQuery configuration
BQ_PROJECT = "stefanoski-generic-code"
BQ_DATASET = "benchmark_data"
BQ_TABLE = "benchmark_results"

# Test settings
TEST_MODE = False