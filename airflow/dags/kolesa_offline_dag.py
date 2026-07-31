# -*- coding: utf-8 -*-
"""
Пересборка и переобучение — офлайн, БЕЗ единого запроса к kolesa.kz.

Это Airflow-версия `run_all --ml`, но не копия его линейной цепочки: здесь
выражен НАСТОЯЩИЙ граф зависимостей, и независимые ветки идут параллельно.
Ради этого Airflow и нужен — иначе он был бы дорогим cron'ом.

Граф и почему он такой:

    clean ──┬── explore ── label_cards      очередь, потом карточки по ней
            ├── evaluate_detector           метрики: нужен только clean_data
            └── train ──┬── ml_dashboard    графики читают модель цены
                        └── residual ── ml_report   отчёт читает ОБА артефакта

  clean первым: он пересобирает clean_data и подхватывает новые вердикты,
    поэтому всё остальное считается по свежим данным;
  evaluate_detector и train не зависят друг от друга — идут одновременно;
  ml_dashboard читает только модель цены, поэтому ждать калибровку пола ему
    незачем, а вот ml_report требует и модель, и пол.

Безопасно запускать в любой момент: сетевых тасков нет, суточный лимит
запросов не расходуется. Поэтому DAG включён и запускается вручную
(schedule=None) — расписание тут не нужно, пересборка нужна после разметки,
а не по будильнику.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"


def job(task_id: str, module: str) -> BashOperator:
    """Таск = запуск модуля пакета из корня проекта."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && python -m {module}",
    )


with DAG(
    dag_id="kolesa_offline_rebuild",
    description="Пересборка clean_data, отчёты и переобучение моделей (без сети)",
    schedule=None,               # только вручную
    start_date=datetime(2026, 7, 19),
    catchup=False,
    max_active_runs=1,           # два прогона писали бы в одни таблицы
    default_args={"owner": "kolesa-antifraud", "retries": 1},
    tags=["kolesa", "offline", "ml", "safe"],
    # В отличие от сетевого DAG'а этот создаётся ВКЛЮЧЁННЫМ, и это безопасно:
    # schedule=None означает, что сам он не стартует никогда. А выключенный
    # DAG в Airflow не исполнится даже при ручном «Trigger» — прогон повис бы
    # в очереди, что выглядит как поломка.
    is_paused_upon_creation=False,
) as dag:

    clean     = job("clean",             "kz.transform.clean")
    explore   = job("explore",           "kz.report.explore")
    cards     = job("label_cards",       "kz.report.label_cards")
    evaluate  = job("evaluate_detector", "kz.report.evaluate_detector")
    train     = job("train_price_model", "kz.ml.train_price_model")
    residual  = job("residual_detector", "kz.ml.residual_detector")
    dashboard = job("ml_dashboard",      "kz.report.ml_dashboard")
    report    = job("ml_report",         "kz.report.ml_report")

    clean >> explore >> cards
    clean >> evaluate
    clean >> train >> dashboard
    train >> residual >> report
