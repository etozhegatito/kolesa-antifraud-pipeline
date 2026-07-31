# -*- coding: utf-8 -*-
"""
DAG офлайн-пересборки = `run_all.py --fast`: чистка + отчёты, БЕЗ сети.

Зачем отдельно от kolesa_antifraud_pipeline: тот ходит на kolesa.kz, и
запускать его «просто посмотреть, работает ли Airflow» нельзя — каждый
холостой прогон тратит лимит запросов с домашнего IP. Здесь сетевых
тасков нет вообще: читаем raw-слой из Postgres, пересобираем clean_data
и артефакты EDA, пишем обратно в Postgres.

Поэтому этот DAG — правильный первый запуск после поднятия Airflow:
он проверяет ровно то, что нужно проверить (контейнер видит Postgres,
проект примонтирован, права на запись есть), и ничего не стоит.

schedule=None — только ручной запуск: пересборка нужна после разметки
или правок правил, а не по будильнику.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"

with DAG(
    dag_id="kolesa_offline_rebuild",
    description="Офлайн-пересборка clean+отчёты, без запросов к kolesa.kz",
    schedule=None,               # только вручную
    start_date=datetime(2026, 7, 19),
    catchup=False,
    # В отличие от сетевого DAG'а этот создаётся ВКЛЮЧЁННЫМ, и это безопасно:
    # schedule=None означает, что сам он не стартует никогда. А выключенный
    # DAG в Airflow не исполнится даже при ручном «Trigger» — прогон повиснет
    # в очереди, что выглядит как поломка. Включённый + без расписания = ровно
    # «работает только когда я нажму».
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args={"owner": "kolesa-antifraud", "retries": 1},
    tags=["kolesa", "offline", "safe"],
) as dag:

    # Проверка связи с Postgres до тяжёлых шагов: если контейнер не видит
    # базу, лучше упасть здесь с внятной ошибкой, чем внутри clean.py.
    check_db = BashOperator(
        task_id="check_postgres",
        bash_command=(
            f"cd {PROJECT_DIR} && python -c "
            "\"from kz.core.db import get_engine; from sqlalchemy import text; "
            "c=get_engine().connect(); "
            "print('raw_ads:', c.execute(text('SELECT COUNT(*) FROM raw_ads'))"
            ".scalar()); "
            "print('clean_data:', c.execute(text('SELECT COUNT(*) FROM "
            "clean_data')).scalar())\""
        ),
    )
    clean = BashOperator(
        task_id="clean",
        bash_command=f"cd {PROJECT_DIR} && python -m kz.transform.clean",
    )
    explore = BashOperator(
        task_id="explore",
        bash_command=f"cd {PROJECT_DIR} && python -m kz.report.explore",
    )
    cards = BashOperator(
        task_id="label_cards",
        bash_command=f"cd {PROJECT_DIR} && python -m kz.report.label_cards",
    )

    check_db >> clean >> explore >> cards
