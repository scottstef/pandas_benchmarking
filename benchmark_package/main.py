import sys
import time
import random
import numpy as np
from datetime import datetime

# Import package modules
from benchmark_package import config
from benchmark_package.data import generator
from benchmark_package.benchmarks import pandas_bench, dask_bench, spark_bench, python_bench
from benchmark_package.storage import sqlite_manager, bigquery_manager
from benchmark_package.visualization import plotter

def save_benchmark_results(run_date, count, num_cols, file_size_mb, pandas_time, dask_time, spark_time, native_python_time):
    """Save benchmark results to all configured storage systems."""
    # Save to SQLite and get the ID
    row_id = sqlite_manager.save_results(
        run_date, count, num_cols, file_size_mb, 
        pandas_time, dask_time, spark_time, native_python_time
    )
    
    # Save to BigQuery with the same ID
    bigquery_manager.save_results(
        row_id, run_date, count, num_cols, file_size_mb,
        pandas_time, dask_time, spark_time, native_python_time
    )

def main():
    """Main benchmark execution function."""
    # Print system information
    print(f'Source is {config.SOURCE}')
    
    # Initialize databases
    sqlite_manager.initialize_database()
    print('SQLite database initialized')
    
    bigquery_manager.initialize_bigquery()
    print('BigQuery table initialized')

    # Initialize SparkSession
    spark = spark_bench.create_spark_session()

    # Set random seed for reproducibility
    seed_value = int(time.time())
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # Determine number of tests to run
    num_tests = 1 if config.TEST_MODE else 20

    for i in range(num_tests):
        print(f"\n=== Run {i+1} ===")
        
        # Determine test parameters
        if config.TEST_MODE:
            count = 1
            num_cols = 10
        else:
            count = random.randint(1, 30)
            num_cols = random.randint(1, 20)

        print(f"Running with count={count} hundred thousand rows and num_cols={num_cols} columns")

        # Generate test data
        file_size_mb = generator.generate_data(count, num_cols)

        # Run all benchmarks
        pandas_time = pandas_bench.run_benchmark()
        dask_time = dask_bench.run_benchmark()
        spark_time = spark_bench.run_benchmark(spark)
        native_python_time = python_bench.run_benchmark()

        # Save results
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_benchmark_results(
            run_date, count, num_cols, file_size_mb,
            pandas_time, dask_time, spark_time, native_python_time
        )

    # Visualize results
    plotter.plot_benchmarks()
    
    # Show BigQuery results summary
    bigquery_manager.query_results()

    # Clean up
    spark.stop()

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        config.TEST_MODE = True
    
    main()