# -*- coding: utf-8 -*-
"""Offline cards for manual market-anomaly verdicts.

Review requires enough visual and textual evidence to understand a vehicle.
Opening every listing on kolesa.kz is unreliable for archived pages and adds
untracked traffic from the same IP as collection jobs. Instead, the project
builds cards from data already stored locally and loads only image-CDN assets.

Use ``python -m kz.web`` for the canonical application: verdicts live at
``/label`` and photo damage labelling at ``/damage``. Use
``python -m kz.report.label_cards`` only for an offline HTML export. The
``--rule-only`` option is a narrow rule-detector diagnostic.

Selections have three persistence layers: the current browser session,
``localStorage`` for reload recovery, and ``data/manual_labels.csv`` as the
source of truth. A ``file://`` export cannot write the journal.

The package is split by responsibility: ``queue.py`` selects rows,
``render.py`` builds HTML without database or disk writes, and ``journal.py``
is the only verdict-mutation layer. HTTP writes are handled by ``kz.web.app``.
Names below are re-exported for ``from kz.report import label_cards``.
"""

from kz.report.label_cards.journal import (  # noqa: F401
    BASE_HEADER,
    LABELS_CSV,
    LABELS_PREV,
    STRATUM_COLS,
    VERDICTS,
    dedupe_journal,
    journal_facts,
    journal_header,
    read_journal,
    upsert_verdict,
    write_journal,
)
from kz.report.label_cards.queue import QUEUE_CSV, load_rows  # noqa: F401
from kz.report.label_cards.render import (  # noqa: F401
    DEAD_HOSTS,
    FLAG_HELP,
    OUT_HTML,
    PRICE_BANDS,
    build,
    card_html,
    fmt,
    money,
    photo_src,
    price_band,
    price_verdict_hint,
)
