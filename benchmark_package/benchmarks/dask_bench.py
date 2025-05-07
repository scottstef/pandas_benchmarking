import time
import dask.dataframe as dd
from benchmark_package import config
from benchmark_package.benchmarks import benchmark_results

def run_benchmark():
    """Run benchmark using Dask.
    
    Returns:
        float: Execution time in seconds
    """
    print("\n=== Dask Benchmark ===")
    start = time.time()

    df_dask = dd.read_csv(config.CSV_PATH)
    result_dask = df_dask.groupby('col_0').mean().compute()

    end = time.time()
    dask_time = end - start
    print(f"Dask time: {dask_time:.2f} seconds")
    benchmark_results.append(('Dask', dask_time))

    del df_dask
    del result_dask
    return dask_time