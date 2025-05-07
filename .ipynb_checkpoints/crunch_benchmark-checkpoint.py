import os
import time
import sqlite3
from datetime import datetime
import random
import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import csv

csv_path = 'fake_data.csv'
db_path = 'benchmark_results.db'
benchmark_results = []

# Function to generate data
def generate_data(count, num_cols):
    NUM_ROWS = count * 1_000_000  # 1 million rows per "count"
    NUM_COLS = num_cols

    print("\nGenerating fake data with pandas...")
    data = np.random.rand(NUM_ROWS, NUM_COLS)
    columns = [f'col_{i}' for i in range(NUM_COLS)]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV using Pandas
    print('Starting to write file')
    df.to_csv(csv_path, index=False)
    print('Done writing to file')

    file_stats = os.stat(csv_path)
    file_size_mb = file_stats.st_size / (1024 * 1024)  # File size in MB
    print(f'File Size in MegaBytes is {file_size_mb:.2f}')
    return file_size_mb  

# Benchmark using Pandas
def pandas_benchmark():
    print("\n=== Pandas Benchmark ===")
    start = time.time()

    df_pandas = pd.read_csv(csv_path)
    result_pandas = df_pandas.groupby('col_0').mean()

    end = time.time()
    pandas_time = end - start
    print(f"Pandas time: {pandas_time:.2f} seconds")
    benchmark_results.append(('Pandas', pandas_time))

    del df_pandas
    del result_pandas
    return pandas_time

# Benchmark using Dask
def dask_benchmark():
    print("\n=== Dask Benchmark ===")
    start = time.time()

    df_dask = dd.read_csv(csv_path)
    result_dask = df_dask.groupby('col_0').mean().compute()

    end = time.time()
    dask_time = end - start
    print(f"Dask time: {dask_time:.2f} seconds")
    benchmark_results.append(('Dask', dask_time))

    del df_dask
    del result_dask
    return dask_time

# Native Python Benchmark (mean calculation with native Python)
def native_python_benchmark():
    print("\n=== Native Python Benchmark ===")
    start = time.time()

    # Read the CSV file using Python's built-in csv module
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header row
        col_0 = [float(row[0]) for row in reader]
        col_1 = [float(row[1]) for row in reader]  # Assuming another column for the mean calculation

    # Group by col_0 and calculate the mean for col_1 (simulating groupby in pandas/dask)
    grouped_data = {}
    for value, col_1_val in zip(col_0, col_1):
        if value not in grouped_data:
            grouped_data[value] = []
        grouped_data[value].append(col_1_val)

    # Calculate the mean of each group
    means = {key: sum(values) / len(values) for key, values in grouped_data.items()}

    end = time.time()
    native_python_time = end - start
    print(f"Native Python time: {native_python_time:.2f} seconds")
    benchmark_results.append(('Native Python', native_python_time))

    return native_python_time


# Initialize the database
def initialize_database():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            count INTEGER,
            num_cols INTEGER,
            file_size_mb REAL,
            pandas_time REAL,
            dask_time REAL,
            native_python_time REAL
        )
    ''')
    conn.commit()
    conn.close()

# Save benchmark results into the database
def save_results(run_date, count, num_cols, file_size_mb, pandas_time, dask_time, native_python_time):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        INSERT INTO benchmarks (run_date, count, num_cols, file_size_mb, pandas_time, dask_time, native_python_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (run_date, count, num_cols, file_size_mb, pandas_time, dask_time, native_python_time))
    conn.commit()
    conn.close()

# Plot the benchmark results
def plot_benchmarks():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file_name = f"benchmark_results_{timestamp}.png"

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM benchmarks", conn)
    conn.close()

    plt.figure(figsize=(14, 8))
    plt.plot(df['run_date'], df['pandas_time'], label='Pandas', marker='o')
    plt.plot(df['run_date'], df['dask_time'], label='Dask', marker='o')
    plt.plot(df['run_date'], df['native_python_time'], label='Native Python', marker='o')

    for i, row in df.iterrows():
        annotation = f"{row['count']}M, {row['num_cols']} cols"
        plt.annotate(annotation, (row['run_date'], row['pandas_time']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')
        plt.annotate(annotation, (row['run_date'], row['dask_time']),
                     textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='blue')
        plt.annotate(annotation, (row['run_date'], row['native_python_time']),
                     textcoords="offset points", xytext=(0,-30), ha='center', fontsize=8, color='red')

    plt.xlabel('Run Date')
    plt.ylabel('Time (seconds)')

    last_run = df['run_date'].max()
    total_runs = len(df)
    plt.title(f'Benchmark Performance Over Time\n{total_runs} Runs • Last Test: {last_run}', fontsize=16)

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(1.0, -0.15, f"Plot generated: {timestamp_now}",
             ha='right', va='center', transform=plt.gca().transAxes, fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(plot_file_name, dpi=300)
    print(f"Plot saved as {plot_file_name}.")

    plt.show(block=False)

# Main function
def main():
    initialize_database()

    seed_value = int(time.time())  # Using current time as a dynamic seed for randomness
    random.seed(seed_value)        # Seed the random module
    np.random.seed(seed_value)

    for i in range(2):  # Run 20 tests
        print(f"\n=== Run {i+1} ===")

        count = random.randint(1, 30)  # Randomly choose a count (number of millions of rows)
        num_cols = random.randint(1, 40)  # Randomly choose the number of columns

        print(f"Running with count={count} million rows and num_cols={num_cols} columns")

        file_size_mb = generate_data(count, num_cols)

        # Run Pandas Benchmark
        pandas_time = pandas_benchmark()

        # Run Dask Benchmark
        dask_time = dask_benchmark()

        # Run Native Python Benchmark
        native_python_time = native_python_benchmark()

        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_results(run_date, count, num_cols, file_size_mb, pandas_time, dask_time, native_python_time)

    # After all runs, plot the results
    plot_benchmarks()

if __name__ == "__main__":
    main()
