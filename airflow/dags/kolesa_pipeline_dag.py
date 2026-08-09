# -*- coding: utf-8 -*-
"""
Data collection from kolesa.kz — the only DAG that touches the network.

This is the Airflow counterpart of `run_all --collect`, and the task order is
the same:

    db_snapshot -> parse_listing -> fill_gaps -> fetch_photos -> clean
                -> explore -> label_cards -> report_new_rows -> report_backlog

  db_snapshot        record current row counts, so the run can report at the
                     end how much data actually arrived;
  parse_listing      fresh listing pages: new ads and price observations;
  fill_gaps          top up statuses, page enrichment and photo hashes in
                     small batches, staying inside the daily request budget;
  fetch_photos       download images to disk while the links are still alive.
                     This is collection, not feature engineering: one of the
                     two CDN hosts was decommissioned and took the photos of
                     1610 ads with it, so the raw files are worth having even
                     though the features derived from them did not help;
  clean / explore    rebuild the clean layer and reports offline, so whatever
                     was collected is reflected in the flags right away;
  label_cards        regenerate the labelling cards for the fresh list;
  report_new_rows    print the delta per table: ads parsed, rows stored;
  report_backlog     what is still missing — statuses, enrichment, hashes —
                     so the next run's size is a number, not a guess.

Why gap filling is a single task instead of one task per job
------------------------------------------------------------
Splitting statuses, enrichment and photos into separate tasks would look
nicer in the UI, but Airflow would then be free to schedule them however it
likes, including in parallel. They all talk to the same host from the same IP,
so the daily request budget would stop being respected. `catch_up` is the only
component that tracks that budget, so it owns the whole top-up and keeps the
jobs strictly sequential. Airflow handles scheduling and observability here,
not request pacing.

The DAG is created paused on purpose. The schedule below is a starting point,
not an instruction: an unattended run would scrape without anyone watching, so
enabling it is a deliberate click in the UI.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"


def job(task_id: str, module: str, *args: str) -> BashOperator:
    """One task = one package module, run from the project root.

    Module stdout becomes the Airflow task log, so whatever the job prints
    (rows written, pages fetched, batches skipped) is visible per task.
    """
    extra = (" " + " ".join(args)) if args else ""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && python -m {module}{extra}",
    )


with DAG(
    dag_id="kolesa_collect",
    description="Collect data from kolesa.kz within the daily request budget",
    schedule="0 9 * * *",
    start_date=datetime(2026, 7, 19),
    catchup=False,          # do not backfill missed days
    max_active_runs=1,      # two concurrent runs would double the request rate
    default_args={"owner": "kolesa-antifraud", "retries": 1},
    tags=["kolesa", "network", "collection"],
    is_paused_upon_creation=True,
) as dag:

    snapshot = job("db_snapshot",     "kz.ops.db_stats", "--save")
    listing  = job("parse_listing",   "kz.collect.parser")
    gaps     = job("fill_gaps",       "kz.ops.catch_up", "--run")
    photos   = job("fetch_photos",    "kz.collect.photo_fetch")
    clean    = job("clean",           "kz.transform.clean")
    explore  = job("explore",         "kz.report.explore")
    cards    = job("label_cards",     "kz.report.label_cards")
    delta    = job("report_new_rows", "kz.ops.db_stats", "--diff")
    state    = job("report_backlog",  "kz.ops.pipeline_status")

    # Photos come after the gap filling but live on a CDN, a different host,
    # so they do not compete for the listing site's quota. Deriving features
    # from them is deliberately not a pipeline step: measurement showed the
    # features do not improve price prediction, and running a network over
    # thousands of images on every collection would buy nothing.
    snapshot >> listing >> gaps >> photos
    photos >> clean >> explore >> cards >> delta >> state
