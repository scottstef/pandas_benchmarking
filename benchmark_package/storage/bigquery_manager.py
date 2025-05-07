from google.cloud import bigquery
from benchmark_package import config
import os


def initialize_bigquery():
    """Initialize BigQuery dataset and table."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/sstefanoski/.config/gcloud/application_default_credentials.json"

    client = bigquery.Client(project=config.BQ_PROJECT)
    
    # Check if dataset exists, create if it doesn't
    try:
        client.get_dataset(f"{config.BQ_PROJECT}.{config.BQ_DATASET}")
        print(f"Dataset {config.BQ_DATASET} already exists.")
    except Exception:
        # Dataset does not exist, create it
        dataset = bigquery.Dataset(f"{config.BQ_PROJECT}.{config.BQ_DATASET}")
        dataset.location = "US"
        dataset = client.create_dataset(dataset)
        print(f"Dataset {config.BQ_DATASET} created.")
    
    # Define schema for the benchmark table
    schema = [
        bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("run_date", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("source", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("count", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("num_cols", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("file_size_mb", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("pandas_time", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("dask_time", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("spark_time", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("native_python_time", "FLOAT", mode="REQUIRED")
    ]
    
    # Create table if it doesn't exist
    table_id = f"{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}"
    try:
        client.get_table(table_id)
        print(f"Table {config.BQ_TABLE} already exists.")
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        table = client.create_table(table)
        print(f"Table {config.BQ_TABLE} created.")

def save_results(row_id, run_date, count, num_cols, file_size_mb, pandas_time, dask_time, spark_time, native_python_time):
    """Save benchmark results to BigQuery.
    
    Args:
        row_id: ID from SQLite to maintain consistency
        run_date: Timestamp of the benchmark run
        count: Number of 100,000 row chunks used
        num_cols: Number of columns used
        file_size_mb: Size of the data file in MB
        pandas_time: Execution time for pandas in seconds
        dask_time: Execution time for dask in seconds
        spark_time: Execution time for spark in seconds
        native_python_time: Execution time for native python in seconds
    """
    client = bigquery.Client(project=config.BQ_PROJECT)
    table_id = f"{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}"
    
    # Format the data for BigQuery
    rows_to_insert = [{
        "id": row_id,
        "run_date": run_date,
        "source": config.SOURCE,
        "count": count,
        "num_cols": num_cols,
        "file_size_mb": file_size_mb,
        "pandas_time": pandas_time,
        "dask_time": dask_time,
        "spark_time": spark_time,
        "native_python_time": native_python_time
    }]
    
    # Insert the data
    errors = client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        print(f"Errors inserting into BigQuery: {errors}")
    else:
        print("Data successfully inserted into BigQuery")

def query_results(group_by_source=True):
    """Query and display benchmark averages from BigQuery.
    
    Args:
        group_by_source: If True, group results by source. If False, show overall averages.
    """
    client = bigquery.Client(project=config.BQ_PROJECT)
    
    if group_by_source:
        query = f"""
        SELECT 
            source,
            COUNT(*) as total_tests,
            AVG(pandas_time) as avg_pandas_time,
            AVG(dask_time) as avg_dask_time,
            AVG(spark_time) as avg_spark_time,
            AVG(native_python_time) as avg_python_time
        FROM 
            `{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}`
        GROUP BY 
            source
        ORDER BY 
            source
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        print("\n=== Benchmark Averages by Source ===")
        for row in results:
            print(f"Source: {row.source} | Tests: {row.total_tests}")
            print(f"  Avg Times (s): Pandas={row.avg_pandas_time:.2f}, Dask={row.avg_dask_time:.2f}, " +
                  f"Spark={row.avg_spark_time:.2f}, Python={row.avg_python_time:.2f}\n")
    else:
        query = f"""
        SELECT 
            COUNT(*) as total_tests,
            AVG(pandas_time) as avg_pandas_time,
            AVG(dask_time) as avg_dask_time,
            AVG(spark_time) as avg_spark_time,
            AVG(native_python_time) as avg_python_time
        FROM 
            `{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_TABLE}`
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        print("\n=== Overall Benchmark Averages ===")
        for row in results:  # There will be only one row in this case
            print(f"Total Tests: {row.total_tests}")
            print(f"  Avg Times (s): Pandas={row.avg_pandas_time:.2f}, Dask={row.avg_dask_time:.2f}, " +
                  f"Spark={row.avg_spark_time:.2f}, Python={row.avg_python_time:.2f}\n")