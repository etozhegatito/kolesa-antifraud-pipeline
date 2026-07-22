# -*- coding: utf-8 -*-
"""
DAG-версия run_all.py — тот же порядок джобов, что и в
`FULL` списке run_all.py, один в один. Это demo-слой (портфолио):
основной способ запуска пайплайна — по-прежнему run_all.py + cron,
см. README. Здесь просто показано, как та же логика выражается через
Airflow, если джобов станет много/нужны отдельные retry и backfill
по каждому шагу.

PROJECT_DIR примонтирован в контейнер Airflow (см. docker-compose.yaml,
профиль "airflow") — BashOperator реально исполняет те же .py-файлы,
ничего не переписано.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"

default_args = {
    "owner": "kolesa-antifraud",
    "retries": 1,
}

with DAG(
    dag_id="kolesa_antifraud_pipeline",
    description="Ежедневный сбор kolesa.kz + антифрод-детекция (= run_all.py)",
    schedule_interval="0 9 * * *",
    start_date=datetime(2026, 7, 19),
    catchup=False,          # не досчитывать пропущенные дни при первом деплое
    default_args=default_args,
    tags=["kolesa", "antifraud", "scraping"],
) as dag:

    def job(task_id: str, script: str) -> BashOperator:
        return BashOperator(
            task_id=task_id,
            bash_command=f"cd {PROJECT_DIR} && python {script}",
        )

    # Порядок и состав — точная копия run_all.py, включая параллельную
    # ветку: enrich (kolesa.kz) и photo_dedup (CDN kcdn.kz) идут
    # одновременно — разные хосты, вежливость per-host не страдает.
    t_parser       = job("parser",        "parser.py")
    t_check_status = job("check_status",  "check_status.py")
    t_clean_pass1  = job("clean_pass_1",  "clean.py")
    t_enrich       = job("enrich",        "enrich.py")
    t_photo_dedup  = job("photo_dedup",   "photo_dedup.py")
    t_clean_pass2  = job("clean_pass_2",  "clean.py")
    t_explore      = job("explore",       "explore.py")

    t_parser >> t_check_status >> t_clean_pass1
    t_clean_pass1 >> [t_enrich, t_photo_dedup] >> t_clean_pass2 >> t_explore
