import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_output_exists(spark, database, table_name, run_id):
    """
    Check if merged output already exists in Hive table to ensure idempotency. 
    Compatible with Spark 3.4.1 and Hive 3.1.0.
    
    Args:
        spark: SparkSession
        database:  Hive database name
        table_name: Hive table name
        run_id: Unique run identifier
    
    Returns: 
        Boolean indicating if data exists for this run_id
    """
    try:
        full_table_name = f"{database}.{table_name}"
        
        # Check if table exists using Spark 3.4.1 catalog API
        if not spark.catalog.tableExists(database, table_name):
            logger.info(f"Table {full_table_name} does not exist. Will create it.")
            return False
        
        # Check if data exists for this run_id
        result = spark.sql(f"""
            SELECT COUNT(*) as cnt 
            FROM {full_table_name} 
            WHERE run_id = '{run_id}'
        """)
        
        existing_data = result.first()['cnt']
        
        if existing_data > 0:
            logger.warning(f"Data for run_id {run_id} already exists in {full_table_name}. Skipping merge.")
            return True
        
        return False
        
    except Exception as e:
        logger. error(f"Error checking table existence: {e}")
        return False


def merge_model_outputs(spark, database, model1_table, model2_table, merged_table, run_id):
    """
    Merge outputs from two model scoring jobs from Hive tables.
    Compatible with Spark 3.4.1 and Hive 3.1.0.
    
    Args:
        spark: SparkSession
        database: Hive database name
        model1_table:  Model 1 output table name
        model2_table: Model 2 output table name
        merged_table: Merged output table name
        run_id:  Unique run identifier for idempotency
    """
    full_merged_table = f"{database}.{merged_table}"
    
    # Check if output already exists (idempotency)
    if check_output_exists(spark, database, merged_table, run_id):
        logger.info("Merge already completed. Exiting.")
        return
    
    # Load model outputs from Hive tables
    model1_full_table = f"{database}.{model1_table}"
    model2_full_table = f"{database}.{model2_table}"
    
    logger.info(f"Loading model 1 output from {model1_full_table} (run_id={run_id})")
    model1_df = spark.sql(f"""
        SELECT * FROM {model1_full_table}
        WHERE run_id = '{run_id}'
    """)
    
    logger.info(f"Loading model 2 output from {model2_full_table} (run_id={run_id})")
    model2_df = spark.sql(f"""
        SELECT * FROM {model2_full_table}
        WHERE run_id = '{run_id}'
    """)
    
    # Validate data exists
    model1_count = model1_df.count()
    model2_count = model2_df.count()
    
    if model1_count == 0:
        raise ValueError(f"No data found in {model1_full_table} for run_id={run_id}")
    if model2_count == 0:
        raise ValueError(f"No data found in {model2_full_table} for run_id={run_id}")
    
    logger.info(f"Model 1 records: {model1_count}")
    logger.info(f"Model 2 records: {model2_count}")
    
    # Determine join key (assuming 'id' column exists)
    join_key = 'id'
    
    if join_key not in model1_df.columns or join_key not in model2_df.columns:
        raise ValueError(f"Join key '{join_key}' not found in both dataframes")
    
    # Select relevant columns from each model
    model1_predictions = model1_df.select(
        join_key,
        col('model_1_prediction').alias('model_1_score'),
        col('scoring_timestamp').alias('model_1_timestamp')
    )
    
    model2_predictions = model2_df.select(
        join_key,
        col('model_2_prediction').alias('model_2_score'),
        col('scoring_timestamp').alias('model_2_timestamp')
    )
    
    # Merge outputs by joining on the key
    logger.info(f"Merging model outputs on key: {join_key}")
    merged_df = model1_predictions.join(model2_predictions, on=join_key, how='inner')
    
    # Add ensemble prediction (example: average of both models)
    # Customize this logic based on your requirements
    merged_df = merged_df.withColumn(
        'ensemble_score',
        (col('model_1_score') + col('model_2_score')) / 2. 0
    )
    
    # Add metadata
    merged_df = merged_df.withColumn('merge_timestamp', current_timestamp()) \
                         .withColumn('run_id', lit(run_id))
    
    # Create database if not exists
    spark. sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    
    # Write to Hive table with Spark 3.4.1 and Hive 3.1.0 compatibility
    if not spark.catalog.tableExists(database, merged_table):
        logger.info(f"Creating Hive table {full_merged_table}")
        merged_df.write \
            .format('hive') \
            .mode('overwrite') \
            .option('fileFormat', 'parquet') \
            .option('compression', 'snappy') \
            .partitionBy('run_id') \
            .saveAsTable(full_merged_table)
        logger.info(f"Created and populated table {full_merged_table}")
    else:
        # Append to existing table with partition overwrite
        logger.info(f"Writing to existing Hive table {full_merged_table}")
        
        # Set dynamic partition overwrite mode for Spark 3.4.1
        spark. conf.set("spark.sql. sources.partitionOverwriteMode", "dynamic")
        
        # Write with overwrite mode and partitionBy for idempotency
        merged_df.write \
            .format('hive') \
            .mode('overwrite') \
            .option('fileFormat', 'parquet') \
            .option('compression', 'snappy') \
            .partitionBy('run_id') \
            .insertInto(full_merged_table, overwrite=True)
        
        logger. info(f"Written data to table {full_merged_table}")
    
    record_count = merged_df.count()
    logger.info(f"Merge completed successfully. Records merged: {record_count}")
    logger.info(f"Data written to: {full_merged_table} (partition: run_id={run_id})")
    
    # Refresh table metadata
    spark.catalog.refreshTable(full_merged_table)
    logger.info(f"Refreshed table metadata for {full_merged_table}")


def main():
    parser = argparse.ArgumentParser(description='Spark 3.4.1 Output Merger with Hive 3.1.0')
    parser.add_argument('--database', required=True, help='Hive database name')
    parser.add_argument('--model1-table', required=True, help='Model 1 output table name')
    parser.add_argument('--model2-table', required=True, help='Model 2 output table name')
    parser.add_argument('--merged-table', required=True, help='Merged output table name')
    parser.add_argument('--run-id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Create Spark session with Hive 3.1.0 support
    spark = SparkSession.builder \
        .appName("Model Output Merger") \
        .config("spark.sql.hive. version", "3.1.0") \
        .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive. exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition. mode", "nonstrict") \
        .config("spark.sql.hive.convertMetastoreParquet", "true") \
        .config("spark.sql. parquet.writeLegacyFormat", "false") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # Log Spark version
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Using Hive metastore")
    
    try:
        logger.info(f"Starting merge for run ID: {args.run_id}")
        logger.info(f"Database: {args.database}")
        
        merge_model_outputs(
            spark=spark,
            database=args.database,
            model1_table=args.model1_table,
            model2_table=args.model2_table,
            merged_table=args.merged_table,
            run_id=args.run_id
        )
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        raise
    finally: 
        spark.stop()


if __name__ == '__main__': 
    main()