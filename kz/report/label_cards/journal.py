# -*- coding: utf-8 -*-
"""Журнал вердиктов: единственное место, где проект пишет ручную разметку.

Правило номер один всего проекта: журнал НИКОГДА не пересоздаётся и не
обрезается, только дополняется и правится по строкам. Один раз разметку уже
чуть не потеряли, пересобрав файл из очереди, — вердикт остаётся валидным,
даже если объявление давно ушло из подозрительных.

Отсюда всё остальное в этом файле: атомарная запись через временный файл,
снимок предыдущего состояния перед первой правкой запуска, чтение и запись
csv-модулем вместо pandas (тот превращает целые в «50.0» при round-trip —
реальный баг, ронявший вставку в Postgres).
"""

import csv
import os
import shutil
from pathlib import Path

import pandas as pd

LABELS_CSV  = "data/manual_labels.csv"
# Состояние журнала до правок текущего запуска — точка восстановления.
# Файл один и перезаписывается, чтобы не разводить гору бэкапов.
LABELS_PREV = "data/manual_labels.prev.csv"

VERDICTS = ("fraud", "legit", "unknown")


# Слой выборки обязан храниться В ЖУРНАЛЕ, а не только в очереди. Очередь —
# список работы, она пересобирается и намеренно выкидывает уже размеченное.
# Из-за этого метаданные терялись: после разметки контрольных выяснить, что
# они были контрольными, стало невозможно, а без этого не оценить пропуски.
STRATUM_COLS = ["sampling_stratum", "stratum_population"]

BASE_HEADER = ["ad_id", "url", "title", "year", "price_tenge", "mileage_km",
               "suspicion_reasons", "seller_comment", "verdict", "comment"]


def journal_header() -> list[str]:
    """Порядок колонок журнала берём из самого файла, а не из константы:
    файл ведётся руками, и его схема — источник истины. Недостающие колонки
    слоя добавляем в конец, чтобы старые журналы продолжали работать."""
    head = None
    if Path(LABELS_CSV).exists():
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            head = next(csv.reader(f), None)
    head = list(head) if head else list(BASE_HEADER)
    for c in STRATUM_COLS:
        if c not in head:
            head.append(c)
    return head


def _cell(v) -> str:
    """Значение для CSV: пропуск → пусто, целое → без «.0».

    Именно из-за «.0» правило проекта запрещает писать журнал через pandas:
    round-trip превращал 50 в "50.0" и ронял вставку в INTEGER-колонку.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v)


_snapshot_done = False


def _snapshot_once() -> None:
    """Один раз за запуск сохранить состояние журнала ДО правок.

    Журнал — ручной ground truth, его нельзя потерять, а он не в git
    (data/ в .gitignore). Поэтому перед первой записью кладём рядом
    предыдущую версию: всегда есть точка восстановления, и при этом файл
    один, а не гора бэкапов.
    """
    global _snapshot_done
    if _snapshot_done:
        return
    _snapshot_done = True
    if Path(LABELS_CSV).exists():
        shutil.copyfile(LABELS_CSV, LABELS_PREV)


def read_journal() -> tuple[list[str], list[dict]]:
    """Журнал как есть, строками-словарями. Читаем csv-модулем: значения
    остаются ровно теми строками, что в файле, ничего не переформатируется."""
    if not Path(LABELS_CSV).exists():
        return journal_header(), []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [dict(x) for x in r]
        return list(r.fieldnames or journal_header()), rows


def write_journal(header: list[str], rows: list[dict]) -> None:
    """Атомарная запись: сначала во временный файл, потом подмена. Так
    журнал не останется обрезанным, если процесс умрёт на середине."""
    Path(LABELS_CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(LABELS_CSV) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in header})
    os.replace(tmp, LABELS_CSV)


def upsert_verdict(ad_id: str, verdict: str, comment: str, facts: dict) -> None:
    """Записать вердикт: строка по этому ad_id уже есть → ОБНОВИТЬ её на
    месте; нет → дописать новую.

    Раньше здесь был чистый append, и повторные нажатия плодили по несколько
    строк на одно объявление с противоречивыми вердиктами (fraud, потом
    legit, потом legit с комментарием). clean.py берёт последнюю, поэтому
    работало верно, но журнал читался как мусор и глазами не проверялся.

    Обновляется ПЕРВАЯ строка по объявлению — она стоит на своём месте из
    очереди разметки, и порядок файла не съезжает. Лишние дубликаты того же
    ad_id при этом убираются: файл сам приходит в порядок по мере разметки.

    Смысл правила «журнал не перезаписывается» сохранён: вердикты не
    теряются, предыдущая версия файла лежит в manual_labels.prev.csv, а
    запись атомарна.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"недопустимый вердикт: {verdict!r}")
    _snapshot_once()
    header, rows = read_journal()
    aid = str(ad_id)
    same = [r for r in rows if str(r.get("ad_id", "")) == aid]
    if same:
        target = same[0]                       # первая — её и правим
        keep = set(id(r) for r in same[1:])     # прочие дубликаты убираем
        rows = [r for r in rows if id(r) not in keep]
    else:
        target = {c: "" for c in header}
        target.update({c: _cell(facts.get(c)) for c in header if c in facts})
        target["ad_id"] = aid
        rows.append(target)
    target["verdict"] = verdict
    target["comment"] = comment or ""
    write_journal(header, rows)


def dedupe_journal() -> tuple[int, int]:
    """Свернуть накопленные дубликаты: одна строка на объявление.

    Побеждает ПОСЛЕДНИЙ непустой вердикт (это и был твой финальный выбор),
    а место в файле сохраняется за ПЕРВОЙ строкой объявления.
    Возвращает (сколько строк было, сколько стало).
    """
    header, rows = read_journal()
    before = len(rows)
    _snapshot_once()
    order, best = [], {}
    for r in rows:
        aid = str(r.get("ad_id", ""))
        if aid not in best:
            order.append(aid)
            best[aid] = dict(r)
            continue
        # непустой вердикт перекрывает; пустой не затирает уже выбранный
        if str(r.get("verdict", "")).strip():
            best[aid]["verdict"] = r["verdict"]
            best[aid]["comment"] = r.get("comment", "")
    out = [best[a] for a in order]
    write_journal(header, out)
    return before, len(out)


def journal_facts(rows: pd.DataFrame) -> dict:
    """ad_id → описательные поля для строки журнала."""
    out = {}
    for _, r in rows.iterrows():
        out[str(r["ad_id"])] = {
            "sampling_stratum": r.get("stratum") or "",
            "url": r.get("url") or f"https://kolesa.kz/a/show/{r['ad_id']}",
            "title": f"{r.get('brand') or ''} {r.get('model') or ''}".strip(),
            "year": r.get("year"),
            "price_tenge": r.get("price_tenge"),
            "mileage_km": r.get("mileage_km"),
            "suspicion_reasons": r.get("suspicion_reasons"),
            "seller_comment": r.get("seller_comment"),
        }
    return out
