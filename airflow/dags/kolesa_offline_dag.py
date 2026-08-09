# -*- coding: utf-8 -*-
"""
Rebuild and retrain — fully offline, no requests to kolesa.kz.

This is the Airflow counterpart of `run_all --ml`, but not a copy of its linear
chain: here the real dependency graph is spelled out, so independent branches
run at the same time. That is the reason to use an orchestrator at all —
otherwise it would just be an expensive cron.

    clean ──┬── explore ── label_cards
            ├── evaluate_detector
            └── train_price_model ──┬── ml_dashboard
                                    └── residual_detector ── ml_report
                                                                 │
                                       report_state <────────────┘

Why the graph looks like this:

  clean runs first because it rebuilds clean_data and picks up any new manual
    verdicts, so everything downstream is computed on fresh data;
  evaluate_detector only needs clean_data, so it does not wait for training;
  ml_dashboard reads the price model, while ml_report needs both the price
    model and the calibrated price floor — hence the different branch depths;
  report_state runs last and logs coverage, backlogs and verdict counts, so a
    finished run says what the state is rather than just "success".

Each task prints its own numbers into the Airflow task log: rows cleaned, how
many ads look suspicious, cross-validated error of the price model, precision
of the rule-based detector. Safe to run at any time — nothing here touches the
network, so no request budget is spent.

Scheduling is intentionally left off (`schedule=None`): a rebuild is something
you want after labelling, not on a timer.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"


def job(task_id: str, module: str, *args: str) -> BashOperator:
    """One task = one package module, run from the project root."""
    extra = (" " + " ".join(args)) if args else ""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && python -m {module}{extra}",
    )


with DAG(
    dag_id="kolesa_offline_rebuild",
    description="Rebuild clean layer, refresh reports and retrain models (offline)",
    schedule=None,          # manual runs only
    start_date=datetime(2026, 7, 19),
    catchup=False,
    max_active_runs=1,      # two runs would write to the same tables
    default_args={"owner": "kolesa-antifraud", "retries": 1},
    tags=["kolesa", "offline", "ml"],
    # Created unpaused, which is safe here: with schedule=None it never starts
    # on its own. A paused DAG would not run even when triggered by hand — the
    # run would sit queued, which looks like a failure.
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
    state     = job("report_state",      "kz.ops.pipeline_status")

    clean >> explore >> cards
    clean >> evaluate
    clean >> train >> dashboard
    train >> residual >> report
    [cards, evaluate, dashboard, report] >> state
