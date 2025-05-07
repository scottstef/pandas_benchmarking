import sqlite3
from benchmark_package import config

def initialize_database():
    """Initialize SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    print("Attempting to create table 'benchmarks'...")
    
    # Check if 'source' column exists, if not alter table to add it
    c.execute("PRAGMA table_info(benchmarks)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'benchmarks' not in columns:
        c.execute('''
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT,
                source TEXT,
                count INTEGER,
                num_cols INTEGER,
                file_size_mb REAL,
                pandas_time REAL,
                dask_time REAL,
                spark_time REAL,
                native_python_time REAL
            )
        ''')
        print("Table 'benchmarks' created with source column.")
    elif 'source' not in columns:
        # Add source column to existing table
        c.execute('ALTER TABLE benchmarks ADD COLUMN source TEXT')
        print("Added 'source' column to existing table.")
    else:
        print("Table 'benchmarks' already exists with source column.")
    
    conn.commit()
    conn.close()

def save_results(run_date, count, num_cols, file_size_mb, pandas_time, dask_time, spark_time, native_python_time):
    """Save benchmark results to SQLite database.
    
    Returns:
        int: ID of the inserted record
    """
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO benchmarks 
        (run_date, source, count, num_cols, file_size_mb, pandas_time, dask_time, spark_time, native_python_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (run_date, config.SOURCE, count, num_cols, file_size_mb, pandas_time, dask_time, spark_time, native_python_time))
    conn.commit()
    
    # Get the ID of the inserted row
    row_id = c.lastrowid
    conn.close()
    return row_id

def get_benchmark_data():
    """Retrieve all benchmark data from SQLite.
    
    Returns:
        pandas.DataFrame: DataFrame with all benchmark data
    """
    import pandas as pd
    conn = sqlite3.connect(config.DB_PATH)
    df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    conn.close()
    return df