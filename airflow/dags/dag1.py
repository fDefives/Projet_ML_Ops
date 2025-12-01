from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime
import os
import subprocess

# Chemins
WATCH_FOLDER = "/opt/airflow/data/pred"  # à adapter
OUTPUT_FOLDER = "/opt/airflow/data/pred"    # dossier de sortie pour segmentation

# Fonction pour vérifier si un nouveau dossier est présent
def check_new_folder(**kwargs):
    folders = [f for f in os.listdir(WATCH_FOLDER) if os.path.isdir(os.path.join(WATCH_FOLDER, f))]
    if folders:
        kwargs['ti'].xcom_push(key='new_folder', value=folders[0])
        return True
    return False

# Fonction pour lancer le script de segmentation Spark
def run_segmentation(**kwargs):
    folder = kwargs['ti'].xcom_pull(key='new_folder')
    subprocess.run(["python3", "segmentation_spark.py", os.path.join(WATCH_FOLDER, folder), OUTPUT_FOLDER], check=True)

# Fonction pour lancer le script de prédiction MLflow
def run_prediction(**kwargs):
    folder = kwargs['ti'].xcom_pull(key='new_folder')
    output_path = os.path.join(OUTPUT_FOLDER, folder)
    subprocess.run(["python3", "predict_mlflow.py", output_path], check=True)

# DAG definition
with DAG(
    dag_id="auto_segmentation_prediction",
    start_date=datetime(2025, 12, 1),
    schedule_interval=None,  # déclenché par le capteur
    catchup=False,
    tags=["automation"]
) as dag:

    sensor_task = PythonSensor(
        task_id="wait_for_new_folder",
        python_callable=check_new_folder,
        poke_interval=60,  # vérifie toutes les minutes
        timeout=3600       # timeout au bout d'une heure
    )

    segmentation_task = PythonOperator(
        task_id="run_segmentation",
        python_callable=run_segmentation
    )

    prediction_task = PythonOperator(
        task_id="run_prediction",
        python_callable=run_prediction
    )

    sensor_task >> segmentation_task >> prediction_task
