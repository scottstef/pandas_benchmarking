import time
import pandas as pd
from benchmark_package import config
from benchmark_package.benchmarks import benchmark_results

def run_benchmark():
    """Run benchmark using pandas.
    
    Returns:
        float: Execution time in seconds
    """
    print("\n=== Pandas Benchmark ===")
    start = time.time()

    df_pandas = pd.read_csv(config.CSV_PATH)
    result_pandas = df_pandas.groupby('col_0').mean()

    end = time.time()
    pandas_time = end - start
    print(f"Pandas time: {pandas_time:.2f} seconds")
    benchmark_results.append(('Pandas', pandas_time))

    del df_pandas
    del result_pandas
    return pandas_time