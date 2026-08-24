# -*- coding: utf-8 -*-
"""Точка входа: python -m kz.report.label_cards"""

import sys
from pathlib import Path

from kz.report.label_cards.journal import (LABELS_CSV, LABELS_PREV,
                                           dedupe_journal, journal_facts)
from kz.report.label_cards.queue import load_rows
from kz.report.label_cards.render import OUT_HTML, build
from kz.report.label_cards.server import serve

def main():
    include_queue = "--all" in sys.argv
    serve_mode = "--serve" in sys.argv

    if "--dedupe" in sys.argv:
        before, after = dedupe_journal()
        print(f"Журнал: {before} строк → {after} (одна на объявление).")
        print(f"Предыдущая версия сохранена в {LABELS_PREV}.")
        print("Дальше пересобери clean-слой: python -m kz.transform.clean")
        return

    rows = load_rows(include_queue)
    if rows.empty:
        print("Нечего размечать: подозрительных нет.")
        return
    page = build(rows, serve_mode)
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(page, encoding="utf-8")

    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_photo = int(rows["photos"].apply(bool).sum())
    print(f"Карточек: {len(rows)} (мёртвых страниц: {n_dead}, "
          f"с фото: {n_photo})")
    print(f"→ {OUT_HTML}")
    print("kolesa.kz не запрашивается — лимит не тратится.")

    if serve_mode:
        serve(page, journal_facts(rows))
        return
    print("\nВыборы сохраняются в браузере и переживают перезагрузку, но в "
          f"журнал ({LABELS_CSV}) отсюда не попадут: страница, открытая как "
          "file://, писать на диск не может.")
    print("Чтобы вердикты дописывались в журнал сразу: "
          "python -m kz.report.label_cards --serve")


if __name__ == "__main__":
    main()
