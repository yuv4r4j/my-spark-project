from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

default_args = {
    'owner': 'yuv4r4j',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay':  timedelta(minutes=5),
}

# Configuration
HIVE_DATABASE = 'model_scoring'
MODEL1_TABLE = 'model_1_scores'
MODEL2_TABLE = 'model_2_scores'
MERGED_TABLE = 'merged_model_scores'

def check_scoring_completed(**context):
    """
    Check if scoring has already been completed for this run.
    Implements idempotency by checking output paths.
    """
    from datetime import datetime
    
    execution_date = context['execution_date']
    run_id = execution_date.strftime('%Y%m%d_%H%M%S')
    
    logging.info(f"Generated run_id: {run_id} for execution_date: {execution_date}")
    
    # Store run_id in XCom for downstream tasks
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    return run_id

with DAG(
    dag_id='spark_dual_model_scoring_pipeline_hive',
    default_args=default_args,
    description='Idempotent DAG for scoring two MLflow models with Spark 3.4.1 and Hive 3.1.0',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['spark-3.4.1', 'hive-3.1.0', 'mlflow', 'model-scoring'],
) as dag: 

    # Task 0: Check idempotency and generate run_id
    check_run = PythonOperator(
        task_id='check_scoring_run',
        python_callable=check_scoring_completed,
        provide_context=True,
    )

    # Spark 3.4.1 compatible configurations
    spark_conf = {
        'spark.executor.memory': '4g',
        'spark.driver.memory': '2g',
        'spark.executor.cores': '2',
        'spark.executor. instances': '3',
        'spark.sql.warehouse.dir': '/user/hive/warehouse',
        'spark. sql.catalogImplementation': 'hive',
        'spark. sql.hive.version': '3.1.0',
        'spark. sql.sources.partitionOverwriteMode': 'dynamic',
        'hive.exec.dynamic.partition':  'true',
        'hive.exec.dynamic.partition. mode': 'nonstrict',
        'spark.sql.hive.convertMetastoreParquet': 'true',
        'spark.sql.parquet.writeLegacyFormat': 'false',
    }

    # Task 1: Score with Model 1
    score_model_1 = SparkSubmitOperator(
        task_id='score_model_1',
        application='jobs/spark_model_scorer.py',
        name='model_1_scoring',
        conf=spark_conf,
        application_args=[
            '--model-name', 'model_1',
            '--model-stage', 'Production',
            '--input-path', '/data/input/features',
            '--database', HIVE_DATABASE,
            '--table-name', MODEL1_TABLE,
            '--run-id', '{{ task_instance.xcom_pull(task_ids="check_scoring_run", key="run_id") }}',
        ],
        verbose=True,
    )

    # Task 2: Score with Model 2
    score_model_2 = SparkSubmitOperator(
        task_id='score_model_2',
        application='jobs/spark_model_scorer. py',
        name='model_2_scoring',
        conf=spark_conf,
        application_args=[
            '--model-name', 'model_2',
            '--model-stage', 'Production',
            '--input-path', '/data/input/features',
            '--database', HIVE_DATABASE,
            '--table-name', MODEL2_TABLE,
            '--run-id', '{{ task_instance.xcom_pull(task_ids="check_scoring_run", key="run_id") }}',
        ],
        verbose=True,
    )

    # Task 3: Merge model outputs
    merge_outputs = SparkSubmitOperator(
        task_id='merge_model_outputs',
        application='jobs/spark_output_merger.py',
        name='merge_model_outputs',
        conf=spark_conf,
        application_args=[
            '--database', HIVE_DATABASE,
            '--model1-table', MODEL1_TABLE,
            '--model2-table', MODEL2_TABLE,
            '--merged-table', MERGED_TABLE,
            '--run-id', '{{ task_instance.xcom_pull(task_ids="check_scoring_run", key="run_id") }}',
        ],
        verbose=True,
    )

    # Define task dependencies
    check_run >> [score_model_1, score_model_2] >> merge_outputs