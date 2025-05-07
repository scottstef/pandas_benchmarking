# test_crunch_benchmark.py

import os
import sqlite3
import pandas as pd
import pytest
from crunch_benchmark import generate_data, pandas_benchmark, dask_benchmark, initialize_database, save_results, db_path, csv_path

def setup_module(module):
    """ Setup: Run once before all tests """
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

def teardown_module(module):
    """ Teardown: Clean up after tests """
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(csv_path):
        os.remove(csv_path)

def test_generate_data():
    size = generate_data(1, 5)  # 1 million rows, 5 columns
    assert size > 0
    assert os.path.exists(csv_path)

def test_initialize_database():
    initialize_database()
    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='benchmarks';")
    table = c.fetchone()
    conn.close()
    assert table is not None

def test_save_results():
    initialize_database()
    run_date = "2025-04-28 12:00:00"
    file_size_mb = 10.5
    pandas_time = 1.23
    dask_time = 1.11
    count = 1000
    num_cols = 5

    save_results(run_date, count, num_cols, file_size_mb, pandas_time, dask_time)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    conn.close()

    assert len(df) == 1
    assert df.iloc[0]['file_size_mb'] == file_size_mb
    assert len(df) == 1

def test_pandas_benchmark():
    generate_data(1, 5)
    time_taken = pandas_benchmark()
    assert time_taken > 0

def test_dask_benchmark():
    generate_data(1, 5)
    time_taken = dask_benchmark()
    assert time_taken > 0
