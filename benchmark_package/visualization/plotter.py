import matplotlib.pyplot as plt
from datetime import datetime
from benchmark_package import config
from benchmark_package.storage import sqlite_manager

def plot_benchmarks():
    """Generate and save benchmark visualization plot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file_name = f"benchmark_results_{timestamp}.png"

    # Get benchmark data from SQLite
    df = sqlite_manager.get_benchmark_data()

    plt.figure(figsize=(14, 8))
    plt.plot(df['run_date'], df['pandas_time'], label='Pandas', marker='o')
    plt.plot(df['run_date'], df['dask_time'], label='Dask', marker='o')
    plt.plot(df['run_date'], df['spark_time'], label='Spark', marker='o')
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

    # Add source information to the plot
    sources = df['source'].unique()
    source_text = f"Sources: {', '.join(sources)}"
    plt.text(1.0, -0.10, source_text,
             ha='right', va='center', transform=plt.gca().transAxes, fontsize=8, color='gray')

    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.text(1.0, -0.15, f"Plot generated: {timestamp_now}",
             ha='right', va='center', transform=plt.gca().transAxes, fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(plot_file_name, dpi=300)
    print(f"Plot saved as {plot_file_name}.")

    plt.show(block=False)