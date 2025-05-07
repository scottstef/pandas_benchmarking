import os
import numpy as np
import pandas as pd
from benchmark_package import config

def generate_data(count, num_cols):
    """Generate random data for benchmarking.
    
    Args:
        count: Number of 100,000 row chunks to generate
        num_cols: Number of columns to generate
        
    Returns:
        file_size_mb: Size of the generated file in MB
    """
    NUM_ROWS = count * 100_000  # 100_000 rows per "count"
    NUM_COLS = num_cols

    print("\nGenerating fake data with pandas...")
    data = np.random.rand(NUM_ROWS, NUM_COLS)
    columns = [f'col_{i}' for i in range(NUM_COLS)]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV using Pandas
    print('Starting to write file')
    df.to_csv(config.CSV_PATH, index=False)
    print('Done writing to file')

    file_stats = os.stat(config.CSV_PATH)
    file_size_mb = file_stats.st_size / (1024 * 1024)  # File size in MB
    print(f'File Size in MegaBytes is {file_size_mb:.2f}')
    return file_size_mb