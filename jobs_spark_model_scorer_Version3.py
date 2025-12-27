import argparse
import logging
from pyspark.sql import SparkSession
from pyspark.sql. functions import col, lit, current_timestamp
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging. INFO)
logger = logging.getLogger(__name__)


def load_mlflow_model(model_name, model_stage):
    """
    Load MLflow model from registry. 
    
    Args:
        model_name: Name of the model in MLflow registry
        model_stage: Stage of the model (Production, Staging, etc.)
    
    Returns:
        Model URI and version
    """
    client = MlflowClient()
    model_version_info = client.get_latest_versions(model_name, stages=[model_stage])
    
    if not model_version_info:
        raise ValueError(f"No model found for {model_name} in stage {model_stage}")
    
    model_version = model_version_info[0]. version
    model_uri = f"models:/{model_name}/{model_version}"
    
    logger.info(f"Loading model: {model_uri}")
    return model_uri, model_version


def check_output_exists(spark, database, table_name, run_id):
    """
    Check if output already exists in Hive table to ensure idempotency.
    Compatible with Spark 3.4.1 and Hive 3.1.0.
    
    Args:
        spark: SparkSession
        database: Hive database name
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
            logger.warning(f"Data for run_id {run_id} already exists in {full_table_name}.  Skipping scoring.")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking table existence: {e}")
        return False


def create_table_if_not_exists(spark, database, table_name, sample_df):
    """
    Create Hive table if it doesn't exist.
    Uses Spark 3.4.1 and Hive 3.1.0 compatible syntax.
    
    Args:
        spark: SparkSession
        database: Hive database name
        table_name: Hive table name
        sample_df:  Sample DataFrame to infer schema
    """
    full_table_name = f"{database}.{table_name}"
    
    if not spark.catalog.tableExists(database, table_name):
        logger.info(f"Creating Hive table {full_table_name}")
        
        # Create table using DataFrameWriter API (Spark 3.4.1 compatible)
        sample_df.write \
            .format('hive') \
            .mode('overwrite') \
            .option('fileFormat', 'parquet') \
            .option('compression', 'snappy') \
            .partitionBy('run_id') \
            .saveAsTable(full_table_name)
        
        logger.info(f"Successfully created table {full_table_name}")


def score_data(spark, model_uri, input_path, database, table_name, model_name, run_id):
    """
    Score data using MLflow model and write to Hive table.
    Compatible with Spark 3.4.1 and Hive 3.1.0.
    
    Args:
        spark: SparkSession
        model_uri:  URI of the MLflow model
        input_path: Path to input data
        database: Hive database name
        table_name: Hive table name
        model_name: Name of the model for metadata
        run_id: Unique run identifier for idempotency
    """
    full_table_name = f"{database}.{table_name}"
    
    # Check if output already exists (idempotency)
    if check_output_exists(spark, database, table_name, run_id):
        logger.info("Scoring already completed.  Exiting.")
        return
    
    # Load input data
    logger.info(f"Loading input data from {input_path}")
    input_df = spark.read.parquet(input_path)
    
    # Load model as Spark UDF (Spark 3.4.1 compatible)
    logger.info(f"Loading model from {model_uri}")
    predict_udf = mlflow.pyfunc.spark_udf(
        spark=spark, 
        model_uri=model_uri,
        result_type='double'  # Explicit result type for Spark 3.4.1
    )
    
    # Get feature columns (exclude any ID or metadata columns)
    feature_cols = [c for c in input_df.columns if c not in ['id', 'timestamp', 'label']]
    
    # Score the data
    logger.info("Scoring data...")
    scored_df = input_df.withColumn(
        f'{model_name}_prediction',
        predict_udf(*[col(c) for c in feature_cols])
    )
    
    # Add metadata columns for tracking
    scored_df = scored_df.withColumn('scoring_timestamp', current_timestamp()) \
                         .withColumn('model_name', lit(model_name)) \
                         .withColumn('run_id', lit(run_id))
    
    # Create database if not exists
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    logger.info(f"Ensured database {database} exists")
    
    # Write to Hive table with Spark 3.4.1 and Hive 3.1.0 compatibility
    if not spark.catalog.tableExists(database, table_name):
        # Create table on first write
        logger.info(f"Creating Hive table {full_table_name}")
        scored_df.write \
            . format('hive') \
            .mode('overwrite') \
            .option('fileFormat', 'parquet') \
            .option('compression', 'snappy') \
            .partitionBy('run_id') \
            .saveAsTable(full_table_name)
        logger.info(f"Created and populated table {full_table_name}")
    else:
        # Append to existing table with partition overwrite
        logger.info(f"Writing to existing Hive table {full_table_name}")
        
        # Set dynamic partition overwrite mode for Spark 3.4.1
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        
        # Write with overwrite mode and partitionBy for idempotency
        scored_df.write \
            .format('hive') \
            .mode('overwrite') \
            .option('fileFormat', 'parquet') \
            .option('compression', 'snappy') \
            .partitionBy('run_id') \
            .insertInto(full_table_name, overwrite=True)
        
        logger.info(f"Written data to table {full_table_name}")
    
    record_count = scored_df.count()
    logger.info(f"Scoring completed successfully. Records processed: {record_count}")
    logger.info(f"Data written to:  {full_table_name} (partition: run_id={run_id})")
    
    # Refresh table metadata
    spark.catalog.refreshTable(full_table_name)
    logger.info(f"Refreshed table metadata for {full_table_name}")


def main():
    parser = argparse.ArgumentParser(description='Spark 3.4.1 Model Scorer with Hive 3.1.0')
    parser.add_argument('--model-name', required=True, help='MLflow model name')
    parser.add_argument('--model-stage', default='Production', help='MLflow model stage')
    parser.add_argument('--input-path', required=True, help='Input data path')
    parser.add_argument('--database', required=True, help='Hive database name')
    parser.add_argument('--table-name', required=True, help='Hive table name')
    parser.add_argument('--run-id', required=True, help='Unique run identifier')
    parser.add_argument('--mlflow-tracking-uri', default='databricks', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    
    # Create Spark session with Hive 3.1.0 support
    spark = SparkSession. builder \
        .appName(f"Model Scoring - {args.model_name}") \
        .config("spark.sql.hive.version", "3.1.0") \
        .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
        .config("spark.sql.catalogImplementation", "hive") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic. partition.mode", "nonstrict") \
        .config("spark.sql.hive.convertMetastoreParquet", "true") \
        .config("spark.sql.parquet.writeLegacyFormat", "false") \
        .enableHiveSupport() \
        .getOrCreate()
    
    # Log Spark and Hive versions
    logger. info(f"Spark version:  {spark.version}")
    logger.info(f"Using Hive metastore")
    
    try:
        # Load model from MLflow
        model_uri, model_version = load_mlflow_model(args.model_name, args. model_stage)
        
        logger.info(f"Scoring with model: {args.model_name} (version: {model_version})")
        logger.info(f"Run ID: {args.run_id}")
        logger.info(f"Output:  {args.database}.{args.table_name}")
        
        # Score the data
        score_data(
            spark=spark,
            model_uri=model_uri,
            input_path=args.input_path,
            database=args.database,
            table_name=args.table_name,
            model_name=args.model_name,
            run_id=args.run_id
        )
        
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        raise
    finally:
        spark.stop()


if __name__ == '__main__':
    main()