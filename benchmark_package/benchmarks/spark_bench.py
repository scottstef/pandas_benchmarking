import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
from benchmark_package import config

def create_spark_session():
    """Create and configure a Spark session.
    
    Returns:
        SparkSession: Configured Spark session
    """
    spark = (SparkSession
            .builder
            .appName("BenchmarkApp")
            .config("spark.driver.memory", "12g")
            .config("spark.executor.memory", "4g")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "2g")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.sql.shuffle.partitions", "10")
            .config("spark.default.parallelism", "4")
            .config("spark.executor.instances", "2")
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC")
            .getOrCreate())
    
    # Set log level to reduce console output
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark

def run_benchmark(spark_session):
    """Run benchmark using PySpark.
    
    Args:
        spark_session: Active SparkSession
        
    Returns:
        float: Execution time in seconds
    """
    print("\n=== PySpark Benchmark ===")
    start = time.time()

    df_spark = spark_session.read.csv(config.CSV_PATH, header=True)
    result_spark = df_spark.groupBy('col_0').agg(mean('col_0'))
    result_spark.collect()  # Force execution

    end = time.time()
    spark_time = end - start
    print(f"PySpark time: {spark_time:.2f} seconds")
    return spark_time