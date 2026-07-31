# -*- coding: utf-8 -*-
"""
Сбор данных — единственный DAG, который ходит в сеть.

Airflow-версия `run_all --collect`, и порядок здесь тот же по той же причине:

    show_budget ── parser ── catch_up ── clean ── explore ── label_cards

  show_budget    сколько запросов уже потрачено сегодня и где пробелы;
                 дёшево, офлайн, зато сразу видно, стоит ли вообще идти в сеть;
  parser         свежий листинг: новые объявления и наблюдения цен;
  catch_up       добор пробелов (статусы, обогащение, фото) ПОРЦИЯМИ под
                 суточным лимитом на хост. Он считает расход и встаёт, когда
                 квота выбрана;
  дальше         офлайн-пересборка, чтобы собранное сразу попало во флаги.

ПОЧЕМУ ИМЕННО ТАК, А НЕ ОТДЕЛЬНЫМИ ТАСКАМИ НА КАЖДЫЙ СЕТЕВОЙ ДЖОБ.
Соблазн расписать статусы, обогащение и фото отдельными тасками велик —
красивее в UI. Но тогда Airflow запускал бы их по своему усмотрению, включая
параллельно, и суточный лимит перестал бы соблюдаться: все три стучатся в
kolesa.kz с одного IP. Именно такая смесь и положила домашний IP 2026-07-23.
Поэтому весь добор отдан catch_up — он один знает бюджет и держит джобы
строго последовательно. Airflow здесь отвечает за расписание и наблюдаемость,
а не за темп запросов.

DAG создаётся ВЫКЛЮЧЕННЫМ. Расписание оставлено заготовкой: незамеченный
автозапуск означал бы скрейпинг без присмотра. Включать — осознанно,
тумблером в UI.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"


def job(task_id: str, module: str, *args: str) -> BashOperator:
    extra = (" " + " ".join(args)) if args else ""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && python -m {module}{extra}",
    )


with DAG(
    dag_id="kolesa_collect",
    description="Сбор с kolesa.kz под суточным лимитом запросов",
    schedule="0 9 * * *",
    start_date=datetime(2026, 7, 19),
    catchup=False,               # не досчитывать пропущенные дни
    max_active_runs=1,           # два прогона удвоили бы частоту запросов
    default_args={"owner": "kolesa-antifraud", "retries": 1},
    tags=["kolesa", "network", "scraping"],
    is_paused_upon_creation=True,
) as dag:

    budget   = job("show_budget",  "kz.ops.pipeline_status")
    parser   = job("parser",       "kz.collect.parser")
    catch_up = job("catch_up",     "kz.ops.catch_up", "--run")
    clean    = job("clean",        "kz.transform.clean")
    explore  = job("explore",      "kz.report.explore")
    cards    = job("label_cards",  "kz.report.label_cards")

    budget >> parser >> catch_up >> clean >> explore >> cards
