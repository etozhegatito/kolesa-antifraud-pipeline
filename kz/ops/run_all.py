# -*- coding: utf-8 -*-
"""Implementation for the `kz.ops.run_all` module."""

import pathlib as _p

_expected = "run_all.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(
        f"ERROR: this code belongs to {_expected}, but the file is named "
        f"{_p.Path(__file__).name}. Files were mixed up during copying."
    )


import subprocess
import sys
import time


def step(name: str, module: str, *args: str):
    """Implement `step`."""
    return (name, [sys.executable, "-m", module, *args])


STEP_PARSER = step("Collect 1 · listings", "kz.collect.parser")
STEP_STATUS = step("Collect 2 · statuses", "kz.collect.check_status")
STEP_ENRICH = step("Collect 3 · enrichment", "kz.collect.enrich")
STEP_PHOTOS = step("Collect 4 · photo dedup", "kz.collect.photo_dedup")


COLLECT_CHAIN = [
    STEP_PARSER,
    step("Collect 2 · budgeted backlog", "kz.ops.catch_up", "--run"),
    #
    #   python -m kz.ml.photo_features
    #   python -m kz.ml.photo_clip
    step("Collect 3 · photos", "kz.collect.photo_fetch"),
]


STEP_CLEAN = step("Clean · clean_data", "kz.transform.clean")
OFFLINE_CHAIN = [
    STEP_CLEAN,
    step("Report · EDA and queue", "kz.report.explore"),
    step("Report · labelling cards", "kz.report.label_cards"),
]


ML_CHAIN = [
    step("ML 1 · data drift", "kz.ml.monitoring"),
    step("ML 2 · price model", "kz.ml.train_price_model"),
    step("ML 3 · MAPE stability", "kz.ml.mape_stability"),
    step("ML 4 · price floor", "kz.ml.residual_detector"),
    step("ML 5 · price interval", "kz.ml.price_interval"),
    step("ML 6 · model charts", "kz.report.ml_dashboard"),
    step("ML 7 · HTML report", "kz.report.ml_report"),
    step("ML 8 · anomaly-rule evaluation", "kz.report.evaluate_detector"),
    step("ML 9 · listing lifetime", "kz.ml.survival"),
]


def run_step(step) -> None:
    """Implement `run_step`."""
    name, cmd = step
    print(f"\n{'═' * 60}\n▶ {name}\n{'═' * 60}")
    t = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"  … {time.time() - t:.0f}s, exit code {rc}")
    if rc != 0:
        print(f"✖ Step '{name}' failed; stopping the pipeline.")
        sys.exit(rc)


def run_parallel(step_a, step_b) -> None:
    """Implement `run_parallel`."""
    (name_a, cmd_a), (name_b, cmd_b) = step_a, step_b
    print(f"\n{'═' * 60}\n▶ {name_a}  ∥  {name_b}  (parallel)\n{'═' * 60}")
    t = time.time()
    proc_a = subprocess.Popen(cmd_a)
    proc_b = subprocess.Popen(cmd_b)
    rc_a, rc_b = proc_a.wait(), proc_b.wait()
    print(f"  … {time.time() - t:.0f}s, exit codes {rc_a}/{rc_b}")
    for name, rc in ((name_a, rc_a), (name_b, rc_b)):
        if rc != 0:
            print(f"✖ Step '{name}' failed; stopping the pipeline.")
            sys.exit(rc)


def main():
    t0 = time.time()
    ml = "--ml" in sys.argv
    collect = "--collect" in sys.argv

    fast = "--fast" in sys.argv or ml
    light = "--light" in sys.argv

    if collect or (not fast and not light):
        for s in COLLECT_CHAIN:
            run_step(s)
    else:
        if not fast:
            run_step(STEP_PARSER)

    for s in OFFLINE_CHAIN:
        run_step(s)
    if ml:
        for s in ML_CHAIN:
            run_step(s)

    mode = (
        "--ml"
        if ml
        else "--collect"
        if collect
        else "--fast"
        if fast
        else "--light"
        if light
        else "full"
    )
    print(f"\n✔ Pipeline ({mode}) completed in {(time.time() - t0) / 60:.1f} min")
    if ml:
        print("  Review: data/eda/ml_report.html, data/eda/ml_dashboard.png")
    else:
        print("  Review: data/eda/label_cards.html, data/eda/dashboard.png")


if __name__ == "__main__":
    main()
