# -*- coding: utf-8 -*-
"""Разметка повреждений по фотографиям: очередь, журнал и страница.

ЗАЧЕМ. Дешёвый сегмент (до 5 млн) даёт 29,8% ошибки при 15-16% у остальных
и держит общий MAPE в одиночку. Табличными признаками он не лечится: цену
старой машины определяет состояние, а состояния в листинге нет. Единственный
оставшийся путь — увидеть его на фотографиях.

Zero-shot до этого не дотянулся: `clip_damaged` неотличим от монетки, потому
что помятое крыло тонет в асфальте и небе. Нарезка на плитки показала, что
дело именно в масштабе — максимум по плиткам поднимает AUC с 0,776 до 0,827,
и при этом ПОРТИТ ржавчину, которая покрывает машину целиком. Значит рамка
вокруг повреждения даст сети чистый сигнал.

Отсюда этот модуль: разметить 200-300 кадров рамками и обучить логистическую
регрессию на векторах CLIP (они сохраняются, 5633 × 512). Это первый в
проекте случай, когда сеть получает НАШИ метки, а не чужие предобученные.

ЧТО ВАЖНО В УСТРОЙСТВЕ

Координаты хранятся ОТНОСИТЕЛЬНЫМИ (0..1), а не в пикселях: картинка в
браузере масштабируется под окно, и абсолютные координаты сломались бы при
другом размере экрана. Пересчёт в пиксели делается при обучении, когда
известен реальный размер файла.

Кропы НЕ сохраняются картинками. Хранится рамка и ссылка на исходник — тот
же принцип «сырьё неизменяемо», что и во всём проекте: оригинал не трогаем,
разметка живёт отдельным слоем. Передумал насчёт границ — поправил четыре
числа, а не пересохранял файлы.

Журнал ведёт себя как data/manual_labels.csv: только дописывается, строка на
кадр обновляется на месте, атомарная запись, снимок перед первой правкой.
Правило номер один распространяется и сюда — это ручной труд, который не
восстановить пересчётом.

Запуск: python -m kz.report.photo_labels          собрать очередь и открыть
        python -m kz.report.photo_labels --stats  что уже размечено
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Путь журнала переопределяется переменной окружения. Нужно не для гибкости,
# а ради безопасности: проверять живой сервер curl-ом, когда он пишет в
# НАСТОЯЩИЙ журнал, — прямой путь к потере разметки. Один раз так и вышло:
# тестовые записи легли рядом с работой пользователя, а уборка `rm` унесла
# и то и другое.
#
#     KZ_LABELS_DIR=/tmp/scratch python -m kz.web
#
# Проверять руками — только так.
_DIR = os.environ.get("KZ_LABELS_DIR", "data")
LABELS_CSV = str(Path(_DIR) / "photo_labels.csv")
LABELS_PREV = str(Path(_DIR) / "photo_labels.prev.csv")

HEADER = ["ad_id", "position", "path", "label", "x1", "y1", "x2", "y2",
          "comment", "labeled_at"]

# Метки. «unclear» нужен обязательно: без него человек вынужден выбирать
# между двумя неверными вариантами, и в данные попадает шум под видом
# уверенного вердикта.
LABELS = {
    "damaged": "видно повреждение — обвести рамкой",
    "intact":  "повреждений не видно",
    "unclear": "не понять (темно, ракурс, обрезано)",
}

# Сколько контрольных кадров подмешивать к «подозрительным». При доле
# повреждённых около процента случайная выборка дала бы две-три позитивные
# метки на три сотни — учиться было бы не на чем. Поэтому стратификация:
# все кадры помеченных объявлений плюс контроль для отрицательного класса.
CONTROL_PER_POSITIVE = 2


def queue(limit: int = 400) -> pd.DataFrame:
    """Что показывать разметчику: помеченные объявления вперёд, плюс контроль.

    Сначала идут кадры объявлений, где повреждение уже заподозрено по тексту
    или бейджу сайта — там выше шанс встретить настоящий положительный
    пример. Контрольные подмешиваются, чтобы модель училась и на «целых»,
    а не только на «битых».
    """
    from kz.core.db import get_engine
    from kz.ml.photo_clip import load_embeddings

    idx, _ = load_embeddings()
    cd = pd.read_sql(
        "SELECT ad_id, damage_keywords, page_status_badge, price_tenge, age "
        "FROM clean_data", get_engine(), dtype={"ad_id": str})
    cd["suspect"] = (
        (cd.damage_keywords.fillna("").str.len() > 0)
        | cd.page_status_badge.fillna("-").str.contains("вар|ход|залож",
                                                        case=False))
    d = idx.merge(cd, on="ad_id", how="left")
    d["suspect"] = d.suspect.fillna(False)

    done = {(r["ad_id"], str(r["position"])) for r in read_journal()[1]}
    d = d[~d.apply(lambda r: (r.ad_id, str(r.position)) in done, axis=1)]

    pos = d[d.suspect]
    n_ctrl = min(len(d[~d.suspect]), max(0, limit - len(pos)),
                 len(pos) * CONTROL_PER_POSITIVE or limit)
    ctrl = d[~d.suspect].sample(n=n_ctrl, random_state=42) if n_ctrl else d.head(0)
    out = pd.concat([pos, ctrl]).head(limit)
    # перемешиваем: иначе разметчик первые сто кадров видит только битые,
    # привыкает и начинает искать повреждения там, где их нет
    return out.sample(frac=1.0, random_state=7).reset_index(drop=True)


# ─── журнал ─────────────────────────────────────────────────────────────────

_snapshot_done = False


def _snapshot_once() -> None:
    """Один раз за запуск сохранить журнал ДО правок — точка восстановления."""
    global _snapshot_done
    if _snapshot_done or not Path(LABELS_CSV).exists():
        _snapshot_done = True
        return
    shutil.copyfile(LABELS_CSV, LABELS_PREV)
    _snapshot_done = True


def read_journal() -> tuple[list[str], list[dict]]:
    """Журнал как есть. csv-модулем, не pandas: тот при round-trip
    превращает целые в «50.0» (реальный баг проекта)."""
    p = Path(LABELS_CSV)
    if not p.exists():
        return list(HEADER), []
    with p.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or HEADER), list(r)


def write_journal(header: list[str], rows: list[dict]) -> None:
    """Атомарная запись: сначала во временный файл, потом подмена."""
    Path(LABELS_CSV).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(LABELS_CSV + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in header})
    os.replace(tmp, LABELS_CSV)


def save_label(ad_id: str, position, path: str, label: str,
               box=None, comment: str = "") -> None:
    """Записать метку кадра. Повторная разметка ОБНОВЛЯЕТ строку, не плодит.

    box — (x1, y1, x2, y2) в долях от размера картинки, либо None для
    кадров без повреждения.
    """
    if label not in LABELS:
        raise ValueError(f"неизвестная метка: {label!r}")
    if label == "damaged" and not box:
        raise ValueError("для «damaged» нужна рамка")
    if box:
        x1, y1, x2, y2 = (float(v) for v in box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"рамка вне картинки или вывернута: {box}")

    _snapshot_once()
    header, rows = read_journal()
    key = (str(ad_id), str(position))
    row = {
        "ad_id": str(ad_id), "position": str(position), "path": path,
        "label": label,
        "x1": f"{box[0]:.4f}" if box else "",
        "y1": f"{box[1]:.4f}" if box else "",
        "x2": f"{box[2]:.4f}" if box else "",
        "y2": f"{box[3]:.4f}" if box else "",
        "comment": comment,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
    }
    for i, r in enumerate(rows):
        if (r.get("ad_id"), str(r.get("position"))) == key:
            rows[i] = row
            break
    else:
        rows.append(row)
    write_journal(header, rows)


def stats() -> dict:
    """Сколько чего размечено — для счётчика и для решения «хватит ли»."""
    _, rows = read_journal()
    out = dict.fromkeys(LABELS, 0)
    for r in rows:
        if r.get("label") in out:
            out[r["label"]] += 1
    out["total"] = len(rows)
    return out


def main():
    if "--stats" in sys.argv:
        s = stats()
        print(f"Размечено кадров: {s['total']}")
        for k, desc in LABELS.items():
            print(f"  {k:9} {s[k]:4}   {desc}")
        need = 200 - s["damaged"]
        print(f"\nДо обучения нужно ~200 с повреждением: "
              f"{'хватает' if need <= 0 else f'ещё {need}'}")
        return

    q = queue()
    print(f"В очереди кадров: {len(q)}   (помеченных объявлений: "
          f"{int(q.suspect.sum())})")
    print(f"Уже размечено: {stats()['total']}")
    print("\nОткрыть разметку:  python -m kz.web  →  http://127.0.0.1:8000/damage")


if __name__ == "__main__":
    main()
