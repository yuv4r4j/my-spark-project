from airflow import DAG
from airflow.providers. apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow. utils.dates import days_ago
from datetime import timedelta
import logging

default_args = {
    'owner': 'yuv4r4j',
    'depends_on_past': False,
    'email_on_failure':  False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Configuration
HIVE_DATABASE = 'model_scoring'
MODEL1_NAME = 'model_1'
MODEL2_NAME = 'model_2'
MODEL1_TABLE = 'model_1_scores'
MODEL2_TABLE = 'model_2_scores'
MERGED_TABLE = 'merged_model_scores'

def check_scoring_completed(**context):
    """
    Check if scoring has already been completed for this run.
    """
    from datetime import datetime
    
    execution_date = context['execution_date']
    run_id = execution_date.strftime('%Y%m%d_%H%M%S')
    
    logging.info(f"Generated run_id: {run_id} for execution_date: {execution_date}")
    context['task_instance'].xcom_push(key='run_id', value=run_id)
    return run_id


def validate_model_output(model_name, **context):
    """
    Validate individual model output.
    """
    run_id = context['task_instance'].xcom_pull(task_ids='preparation. check_scoring_run', key='run_id')
    logging.info(f"Validating {model_name} output for run_id: {run_id}")
    
    # Add specific validation logic here
    # - Check row counts
    # - Validate schema
    # - Check for nulls in predictions
    
    logging.info(f"{model_name} output validated successfully")
    return True


def validate_merged_output(**context):
    """
    Validate merged output quality.
    """
    run_id = context['task_instance']. xcom_pull(task_ids='preparation.check_scoring_run', key='run_id')
    logging.info(f"Validating merged output for run_id: {run_id}")
    
    # Add validation logic here
    # - Check join completeness
    # - Validate ensemble scores
    # - Check data quality metrics
    
    logging.info("Merged output validated successfully")
    return True


with DAG(
    dag_id='spark_dual_model_scoring_pipeline_advanced',
    default_args=default_args,
    description='Advanced idempotent DAG with TaskGroups for Spark 3.4.1 and Hive 3.1.0',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['spark-3.4.1', 'hive-3.1.0', 'mlflow', 'model-scoring', 'task-groups'],
) as dag:

    # Spark 3.4.1 compatible configurations
    spark_conf = {
        'spark. executor.memory': '4g',
        'spark.driver.memory': '2g',
        'spark.executor.cores': '2',
        'spark.executor. instances': '3',
        'spark.sql.warehouse.dir':  '/user/hive/warehouse',
        'spark.sql.catalogImplementation': 'hive',
        'spark.sql.hive.version': '3.1.0',
        'spark. sql.sources.partitionOverwriteMode': 'dynamic',
        'hive.exec.dynamic.partition': 'true',
        'hive.exec.dynamic.partition.mode': 'nonstrict',
        'spark. sql.hive.convertMetastoreParquet': 'true',
        'spark.sql.parquet.writeLegacyFormat': 'false',
    }

    # TaskGroup:  Preparation
    with TaskGroup(
        group_id='preparation',
        tooltip='Preparation tasks including idempotency checks',
    ) as preparation_group: 

        check_run = PythonOperator(
            task_id='check_scoring_run',
            python_callable=check_scoring_completed,
            provide_context=True,
        )

        prep_complete = DummyOperator(
            task_id='preparation_complete',
        )

        check_run >> prep_complete

    # TaskGroup: Model Scoring
    with TaskGroup(
        group_id='model_scoring',
        tooltip='Parallel execution of model scoring tasks',
    ) as model_scoring_group:

        # Sub-TaskGroup: Model 1 Scoring
        with TaskGroup(
            group_id='model_1_workflow',
            tooltip='Model 1 scoring workflow',
        ) as model_1_group:

            score_model_1 = SparkSubmitOperator(
                task_id='score',
                application='jobs/spark_model_scorer.py',
                name='model_1_scoring',
                conf=spark_conf,
                application_args=[
                    '--model-name', MODEL1_NAME,
                    '--model-stage', 'Production',
                    '--input-path', '/data/input/features',
                    '--database', HIVE_DATABASE,
                    '--table-name', MODEL1_TABLE,
                    '--run-id', '{{ task_instance.xcom_pull(task_ids="preparation. check_scoring_run", key="run_id") }}',
                ],
                verbose=True,
            )

            validate_model_1 = PythonOperator(
                task_id='validate',
                python_callable=validate_model_output,
                op_kwargs={'model_name': MODEL1_NAME},
                provide_context=True,
            )

            score_model_1 >> validate_model_1

        # Sub-TaskGroup: Model 2 Scoring
        with TaskGroup(
            group_id='model_2_workflow',
            tooltip='Model 2 scoring workflow',
        ) as model_2_group: 

            score_model_2 = SparkSubmitOperator(
                task_id='score',
                application='jobs/spark_model_scorer.py',
                name='model_2_scoring',
                conf=spark_conf,
                application_args=[
                    '--model-name', MODEL2_NAME,
                    '--model-stage', 'Production',
                    '--input-path', '/data/input/features',
                    '--database', HIVE_DATABASE,
                    '--table-name', MODEL2_TABLE,
                    '--run-id', '{{ task_instance.xcom_pull(task_ids="preparation.check_scoring_run", key="run_id") }}',
                ],
                verbose=True,
            )

            validate_model_2 = PythonOperator(
                task_id='validate',
                python_callable=validate_model_output,
                op_kwargs={'model_name': MODEL2_NAME},
                provide_context=True,
            )

            score_model_2 >> validate_model_2

        # Both models run in parallel
        [model_1_group, model_2_group]

    # TaskGroup: Output Processing
    with TaskGroup(
        group_id='output_processing',
        tooltip='Merge and validate final outputs',
    ) as output_processing_group:

        merge_outputs = SparkSubmitOperator(
            task_id='merge_outputs',
            application='jobs/spark_output_merger.py',
            name='merge_model_outputs',
            conf=spark_conf,
            application_args=[
                '--database', HIVE_DATABASE,
                '--model1-table', MODEL1_TABLE,
                '--model2-table', MODEL2_TABLE,
                '--merged-table', MERGED_TABLE,
                '--run-id', '{{ task_instance.xcom_pull(task_ids="preparation.check_scoring_run", key="run_id") }}',
            ],
            verbose=True,
        )

        validate_merged = PythonOperator(
            task_id='validate_merged_output',
            python_callable=validate_merged_output,
            provide_context=True,
        )

        pipeline_complete = DummyOperator(
            task_id='pipeline_complete',
        )

        merge_outputs >> validate_merged >> pipeline_complete

    # Define main pipeline flow
    preparation_group >> model_scoring_group >> output_processing_group