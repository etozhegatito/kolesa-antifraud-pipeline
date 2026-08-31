# -*- coding: utf-8 -*-
"""Точка входа: python -m kz.report.label_cards"""

import sys
from pathlib import Path

from kz.report.label_cards.journal import (LABELS_CSV, LABELS_PREV,
                                           dedupe_journal)
from kz.report.label_cards.queue import load_rows
from kz.report.label_cards.render import OUT_HTML, build


def run_unified_web() -> None:
    """Совместимый алиас старого ``--serve`` на единое приложение.

    Отдельный сервер на :8765 раньше формировал только правиловую часть
    очереди, а ``kz.web`` — полную очередь со случайным контролем. Два
    интерфейса к одному журналу давали разные числа и разный статистический
    смысл. Теперь сервер и точка записи ровно одни.
    """
    from kz.web.__main__ import main as web_main

    print("Режим --serve перенесён в единое приложение.")
    print("Открой /label для вердиктов и /damage для разметки фотографий.")
    web_main()


def main():
    if "--serve" in sys.argv:
        run_unified_web()
        return

    # Полная очередь — единственный корректный default: random_control нужен
    # для оценки пропусков (recall), residual_candidate — для второго
    # детектора. Узкий набор оставлен только для явной диагностики правил.
    include_queue = "--rule-only" not in sys.argv

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
    page = build(rows, serve_mode=False)
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(page, encoding="utf-8")

    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_photo = int(rows["photos"].apply(bool).sum())
    print(f"Карточек: {len(rows)} (мёртвых страниц: {n_dead}, "
          f"с фото: {n_photo})")
    print(f"→ {OUT_HTML}")
    print("kolesa.kz не запрашивается — лимит не тратится.")

    print("\nВыборы сохраняются в браузере и переживают перезагрузку, но в "
          f"журнал ({LABELS_CSV}) отсюда не попадут: страница, открытая как "
          "file://, писать на диск не может.")
    print("Чтобы вердикты дописывались в журнал сразу: python -m kz.web, "
          "затем открой http://127.0.0.1:8000/label")


if __name__ == "__main__":
    main()
