import time
import csv
from benchmark_package import config
from benchmark_package.benchmarks import benchmark_results

def run_benchmark():
    """Run benchmark using native Python.
    
    Returns:
        float: Execution time in seconds
    """
    print("\n=== Native Python Benchmark ===")
    start = time.time()

    with open(config.CSV_PATH, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header row
        col_0_index = headers.index('col_0')

        # Initialize a dictionary to store data for each group in col_0
        grouped_data = {}

        # Populate the grouped_data dictionary
        for row in reader:
            group_key = float(row[col_0_index])
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            grouped_data[group_key].append(float(row[col_0_index]))

    # Calculate the mean of 'col_0' for each group
    means = {
        key: sum(values) / len(values)
        for key, values in grouped_data.items()
    }

    end = time.time()
    native_python_time = end - start
    print(f"Native Python time: {native_python_time:.2f} seconds")
    benchmark_results.append(('Native Python', native_python_time))

    return native_python_time