# -*- coding: utf-8 -*-
"""Command-line entry point for ``python -m kz.report.label_cards``."""

import sys
from pathlib import Path

from kz.report.label_cards.journal import LABELS_CSV, LABELS_PREV, dedupe_journal
from kz.report.label_cards.queue import load_rows
from kz.report.label_cards.render import OUT_HTML, build


def run_unified_web() -> None:
    """Redirect the legacy ``--serve`` option to the canonical application.

    The former :8765 server used only rule positives, while ``kz.web`` used the
    statistically complete queue with random controls. One journal must have
    one queue definition and one write endpoint.
    """
    from kz.web.__main__ import main as web_main

    print("The legacy --serve mode now starts the unified application.")
    print("Open /label for verdicts and /damage for photo labelling.")
    web_main()


def main():
    if "--serve" in sys.argv:
        run_unified_web()
        return

    # The full queue is the only statistically valid default. Random controls
    # estimate misses, while residual candidates evaluate the second detector.
    include_queue = "--rule-only" not in sys.argv

    if "--dedupe" in sys.argv:
        before, after = dedupe_journal()
        print(f"Journal: {before} rows → {after} (one per listing).")
        print(f"The previous version was saved to {LABELS_PREV}.")
        print("Next, rebuild the clean layer: python -m kz.transform.clean")
        return

    rows = load_rows(include_queue)
    if rows.empty:
        print("There is nothing to label: no candidates were found.")
        return
    page = build(rows, serve_mode=False)
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(page, encoding="utf-8")

    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_photo = int(rows["photos"].apply(bool).sum())
    print(f"Cards: {len(rows)} (closed pages: {n_dead}, with photos: {n_photo})")
    print(f"→ {OUT_HTML}")
    print("No kolesa.kz requests are made; the collection budget is unchanged.")

    print(
        "\nSelections persist in the browser, but this file:// export cannot "
        f"write the journal ({LABELS_CSV})."
    )
    print("For immediate journal writes, run python -m kz.web and open http://127.0.0.1:8000/label")


if __name__ == "__main__":
    main()
