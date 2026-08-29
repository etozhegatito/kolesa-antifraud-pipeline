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
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
          "comment", "labeled_at", "selection_source", "dataset_split",
          "annotator", "label_version", "boxes_json"]

# Новые объявления получают split детерминированно по ad_id. Старые метки
# НЕЛЬЗЯ задним числом объявить holdout: они уже участвовали в экспериментах.
# Они остаются legacy-train, а независимый audit набирается только из новых,
# случайно выбранных объявлений.
AUDIT_PERCENT = 20
AUDIT_PER_QUEUE = 60
LABEL_VERSION = "3"
MAX_BOXES_PER_FRAME = 20


def split_for_ad(ad_id: str) -> str:
    """Стабильный train/audit split, не зависящий от порядка строк."""
    bucket = int(hashlib.sha256(str(ad_id).encode()).hexdigest()[:8], 16) % 100
    return "audit" if bucket < AUDIT_PERCENT else "train"

# Метки. «unclear» нужен обязательно: без него человек вынужден выбирать
# между двумя неверными вариантами, и в данные попадает шум под видом
# уверенного вердикта.
LABELS = {
    "damaged": "удар, вмятина, разбитая деталь — локально, обвести рамкой",
    "wreck":   "серьёзная авария: перёд или зад разрушен, детали оторваны",
    "parts":   "машина разобрана или снят агрегат (двигатель, коробка)",
    "intact":  "ударов и вмятин нет (ржавчина и потёртости — сюда же)",
    "unclear": "не понять (темно, ракурс, обрезано)",
}

# Граница между «повреждением» и «аварией» — операционная, не на глаз.
#
#   можно обвести рамкой   → damaged   свидетельство в одном месте
#   рамка бессмысленна     → wreck     свидетельство — весь кадр
#
# Тот же принцип, что отделил «разобрано»: рамка есть там, где повреждение
# локально. Если у машины нет переднего бампера, решётки и фары, а детали
# лежат на земле, обводить нечего — разрушен весь узел.
#
# Зачем разделять, а не звать всё «повреждением». Во-первых, однородность
# положительного класса: вмятина на крыле и разбитый вдребезги перёд
# выглядят по-разному, и на двухстах метках сеть, обучаясь на их смеси, не
# выучит ни того, ни другого. Во-вторых, разделение даёт ответ на вопрос,
# который иначе не задать: `clip_damaged` показывал AUC 0,776, и неясно
# было, что именно он ловит. Разумно ожидать, что тяжёлые аварии он видит
# (они контрастны и занимают кадр), а мелкие вмятины нет — но проверить
# это можно только на раздельных метках.
#
# В-третьих, для продукта это разные вещи: вмятина значит «дешевле, но
# ездит», авария — «восстановление сопоставимо с ценой».
#
# И главный довод, тот же что и раньше: объединить классы потом можно
# бесплатно, разделить — невозможно.

# Почему ржавчина идёт в «целая», хотя это очевидно не идеальное состояние.
#
# Ржавчину мы уже умеем видеть: zero-shot CLIP даёт по ней AUC 0,881 без
# единой ручной метки. Тратить ручной труд на то, что и так работает, —
# потеря. Разметка нужна ровно для того, чего CLIP не видит: ударов.
#
# И это не вопрос вкуса, а измеренная разница. Нарезка на плитки подняла
# AUC вмятины с 0,776 до 0,827 и УРОНИЛА ржавчину с 0,881 до 0,809:
# вмятина локальна, ржавчина покрывает кузов целиком. Смешав их в один
# положительный класс на двухстах метках, сеть не выучит ни того, ни
# другого.
#
# Информация не теряется: ржавчина пишется в комментарий. Формулировка
# уточнена 29 августа, на 12-м размеченном кадре из 399 — раньше «intact»
# читалось как «повреждений не видно», и разметчик справедливо спотыкался
# на ржавом, но не битом кузове. На двухсотом кадре такая правка означала
# бы, что ранние и поздние метки про разное.

# Почему «разбор» отдельно от «повреждения», а не вместе.
#
# Реальный кадр из очереди: двигатель Hyundai Sonata лежит на брусчатке,
# в объявлении «Запчасқа болады» и бейдж «Аварийная/Не на ходу». Это не
# вмятина, но и не целая машина.
#
# С двумя сотнями меток положительный класс обязан быть ОДНОРОДНЫМ. Помятое
# крыло и двигатель на земле выглядят совершенно по-разному, и сеть, учась
# на их смеси, не выучит ни того, ни другого. Для продукта это тоже разные
# вещи: вмятина значит «дешевле, но ездит», разбор — «это уже не машина».
#
# Решающий довод: объединить метки потом можно бесплатно, разделить —
# невозможно. Свалив всё в «повреждение» сейчас, мы бы потеряли различие
# навсегда.
#
# Рамка для «разбора» не нужна: свидетельство здесь — весь кадр целиком,
# а не участок на нём.

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

    d["dataset_split"] = d.ad_id.map(split_for_ad)
    d["selection_source"] = np.where(
        d.suspect, "text_or_badge", "random_control")

    # Audit выбирается ДО model-rank и текстовой приоритизации. Иначе это
    # был бы ещё один удобный срез active learning, а не случайная проверка.
    audit_pool = d[d.dataset_split == "audit"].copy()
    n_audit = min(AUDIT_PER_QUEUE, limit, len(audit_pool))
    audit = (audit_pool.sample(n=n_audit, random_state=29)
             if n_audit else audit_pool.head(0))
    audit["selection_source"] = "random_audit"

    train = _mark_candidates(d[d.dataset_split == "train"].copy())
    remaining = max(0, limit - len(audit))
    pos = train[train.suspect]
    per_pos = (CONTROL_PER_POSITIVE if _negatives_so_far() < ENOUGH_NEGATIVES
               else 0)
    n_ctrl = min(len(train[~train.suspect]), max(0, remaining - len(pos)),
                 (len(pos) * per_pos) if per_pos else CONTROL_WHEN_ENOUGH)
    ctrl = (train[~train.suspect].sample(n=n_ctrl, random_state=42)
            if n_ctrl else train.head(0))
    out = pd.concat([audit, pos, ctrl]).head(limit)
    # перемешиваем: иначе разметчик первые сто кадров видит только битые,
    # привыкает и начинает искать повреждения там, где их нет
    out = out.sample(frac=1.0, random_state=7).reset_index(drop=True)
    return _body_first(out)


# Сколько ОБЪЯВЛЕНИЙ брать с верхушки ранжирования и по сколько кадров с
# каждого. Считаем объявлениями, а не кадрами: grouped CV считает
# независимыми объявления, и пять кадров одной машины дают одну точку, а не
# пять. На 29 августа положительных кадров 24, но объявлений всего 13 —
# узкое место именно здесь.
RANK_TOP_ADS = 120
FRAMES_PER_AD = 2

# Контроль. Отрицательных примеров уже за три сотни, на обучение хватает с
# запасом, и тратить очередь на новые «целые» смысла мало. Оставляем
# немного, чтобы оценка доли положительных не поехала совсем.
CONTROL_WHEN_ENOUGH = 60
ENOUGH_NEGATIVES = 200


def _negatives_so_far() -> int:
    """Сколько «целых» уже размечено — от этого зависит доля контроля."""
    return stats()["intact"]


def _mark_candidates(d: pd.DataFrame) -> pd.DataFrame:
    """Кандидаты — верхние ОБЪЯВЛЕНИЯ ранжирования, по паре кадров с каждого.

    До этого отбор шёл только по тексту и бейджу сайта, то есть по тому,
    что НАПИСАНО в объявлении. Ранжирование добавляет отбор по тому, как
    кадр ВЫГЛЯДИТ.

    Берём максимум счёта по объявлению, а не сумму: одного убедительного
    кадра достаточно, чтобы объявление стоило показать, а сумма выдвинула
    бы вперёд просто многофотографийные.

    Перемешивание в `queue` сохраняется намеренно: если показать разметчику
    подряд сотню вероятно битых, он привыкнет и начнёт видеть повреждения
    там, где их нет.
    """
    from kz.ml.photo_clip import load_damage_rank

    rank = load_damage_rank()
    if rank is None or d.empty:
        return d
    rank = rank.copy()
    rank["position"] = rank.position.astype(int)
    m = d.merge(rank, on=["ad_id", "position"], how="left")
    m["damage_rank"] = m.damage_rank.fillna(-1.0)

    by_ad = m.groupby("ad_id").damage_rank.max().nlargest(RANK_TOP_ADS)
    pick = (m[m.ad_id.isin(by_ad.index)]
            .sort_values("damage_rank", ascending=False)
            .groupby("ad_id").head(FRAMES_PER_AD).index)
    newly_ranked = pick[~m.loc[pick, "suspect"].to_numpy()]
    m.loc[pick, "suspect"] = True
    m.loc[newly_ranked, "selection_source"] = "model_rank"
    return m.drop(columns=["damage_rank"])


def _body_first(q: pd.DataFrame) -> pd.DataFrame:
    """Кадры без кузова — в конец очереди, а не прочь из неё.

    Салон, подкапотное, колесо крупным планом и фото документов не могут
    показать повреждение кузова, а это 12% снимков. Гонять по ним человека
    полтора часа — впустую.

    Но и выбрасывать нельзя: порог поставлен на глаз, и ошибка фильтра
    молча унесла бы наружные кадры, среди которых мог быть битый. Поэтому
    не выбрасываем, а откладываем: сначала кузов, потом всё остальное.
    Дошёл до конца — размечай дальше, ничего не потеряно.

    Перемешивание внутри каждой части сохраняется: сортировка стабильная.
    """
    from kz.ml.photo_clip import NO_BODY_THRESHOLD, load_no_body

    nb = load_no_body()
    if nb is None or q.empty:
        return q
    m = q.merge(nb, on=["ad_id", "position"], how="left")
    m["_tail"] = (m["clip_no_body"].fillna(-1.0) > NO_BODY_THRESHOLD).astype(int)
    m = m.sort_values("_tail", kind="stable")
    return m.drop(columns=["_tail", "clip_no_body"]).reset_index(drop=True)


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


def _normalise_boxes(boxes) -> list[tuple[float, float, float, float]]:
    """Проверить список относительных рамок и вернуть числа 0..1."""
    if boxes is None:
        return []
    if not isinstance(boxes, (list, tuple)):
        raise ValueError("boxes должен быть списком рамок")
    if len(boxes) > MAX_BOXES_PER_FRAME:
        raise ValueError(f"слишком много рамок: максимум {MAX_BOXES_PER_FRAME}")

    out = []
    for raw in boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"рамка должна содержать четыре координаты: {raw!r}")
        try:
            x1, y1, x2, y2 = (float(v) for v in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(f"координаты рамки должны быть числами: {raw!r}") from e
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"рамка вне картинки или вывернута: {raw}")
        out.append((x1, y1, x2, y2))
    return out


def boxes_from_row(row: dict) -> list[tuple[float, float, float, float]]:
    """Все рамки строки; старые x1..y2 читаются как список из одной рамки."""
    payload = str(row.get("boxes_json") or "").strip()
    if payload:
        try:
            return _normalise_boxes(json.loads(payload))
        except json.JSONDecodeError as e:
            raise ValueError(f"некорректный boxes_json у {row.get('path')}: {e}") from e
    if row.get("x1") not in (None, ""):
        return _normalise_boxes([[row.get(k) for k in ("x1", "y1", "x2", "y2")]])
    return []


def save_label(ad_id: str, position, path: str, label: str,
               box=None, boxes=None, comment: str = "",
               selection_source: str = "manual",
               dataset_split: str = "train", annotator: str | None = None) -> None:
    """Записать метку кадра. Повторная разметка ОБНОВЛЯЕТ строку, не плодит.

    ``boxes`` — список (x1, y1, x2, y2) в долях от картинки. ``box``
    оставлен для совместимости со старыми вызовами и означает одну рамку.
    В CSV все рамки лежат в ``boxes_json``; x1..y2 дублируют первую, чтобы
    старые исследовательские скрипты продолжили работать.
    """
    if label not in LABELS:
        raise ValueError(f"неизвестная метка: {label!r}")
    if box is not None and boxes is not None:
        raise ValueError("передайте box или boxes, но не оба сразу")
    frame_boxes = _normalise_boxes(boxes if boxes is not None
                                   else ([box] if box is not None else []))
    if label == "damaged" and not frame_boxes:
        raise ValueError("для «damaged» нужна хотя бы одна рамка")
    # Рамка РАЗРЕШЕНА при любой метке, обязательна только для «damaged».
    #
    # Запрещать было ошибкой. Разметчик обводил ржавчину и ставил «целая» —
    # рамка молча отбрасывалась, оставался комментарий. Человек считал, что
    # отмечает область, а координаты не сохранялись ни разу.
    #
    # Тихо терять ручной труд нельзя, даже если сейчас не знаешь, что с ним
    # делать. Ржавчину мы и так детектим (AUC 0,881 zero-shot), но области
    # могут пригодиться: например, чтобы проверить, смотрит ли детектор
    # ржавчины туда же, куда человек.
    _snapshot_once()
    header, rows = read_journal()
    # Миграция только при следующей осознанной записи: существующий журнал
    # не переписывается при импорте модуля. Все старые поля и строки остаются.
    header = list(dict.fromkeys([*header, *HEADER]))
    key = (str(ad_id), str(position))
    if dataset_split not in {"train", "audit"}:
        raise ValueError(f"неизвестный dataset_split: {dataset_split!r}")
    first = frame_boxes[0] if frame_boxes else None
    row = {
        "ad_id": str(ad_id), "position": str(position), "path": path,
        "label": label,
        "x1": f"{first[0]:.4f}" if first else "",
        "y1": f"{first[1]:.4f}" if first else "",
        "x2": f"{first[2]:.4f}" if first else "",
        "y2": f"{first[3]:.4f}" if first else "",
        "comment": comment,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
        "selection_source": selection_source,
        "dataset_split": dataset_split,
        "annotator": annotator or os.environ.get("KZ_ANNOTATOR", "sanzhar"),
        "label_version": LABEL_VERSION,
        "boxes_json": (json.dumps([[round(v, 4) for v in b] for b in frame_boxes],
                                  separators=(",", ":"))
                       if frame_boxes else ""),
    }
    for i, r in enumerate(rows):
        if (r.get("ad_id"), str(r.get("position"))) == key:
            rows[i] = row
            break
    else:
        rows.append(row)
    write_journal(header, rows)


def labelled_frames() -> list[dict]:
    """Уже размеченные кадры — для просмотра и правки на странице.

    Нужны потому, что очередь их СОЗНАТЕЛЬНО не содержит: показывать
    заново то, что уже решено, — трата времени. Но передумать человек
    вправе, и «повреждений 21» должно открываться по клику.
    """
    _, rows = read_journal()
    out = []
    for r in rows:
        if r.get("label") not in LABELS:
            continue
        rec = {"ad_id": r.get("ad_id", ""), "position": int(r.get("position") or 0),
               "path": r.get("path", ""), "label": r["label"],
               "comment": r.get("comment", ""),
               "selection_source": r.get("selection_source", "legacy"),
               "dataset_split": r.get("dataset_split", "train") or "train",
               "annotator": r.get("annotator", ""),
               "label_version": r.get("label_version", "1") or "1"}
        boxes = boxes_from_row(r)
        if boxes:
            rec["boxes"] = [list(b) for b in boxes]
            rec |= dict(zip(("x1", "y1", "x2", "y2"), boxes[0]))
        out.append(rec)
    return out


def stats() -> dict:
    """Сколько кадров И независимых объявлений размечено.

    Для интерфейса полезны кадры, для grouped CV — объявления: пять снимков
    одной машины не превращаются в пять независимых наблюдений.
    """
    _, rows = read_journal()
    out = dict.fromkeys(LABELS, 0)
    out["damage_boxes"] = 0
    ads = {label: set() for label in LABELS}
    for r in rows:
        if r.get("label") in out:
            out[r["label"]] += 1
            ads[r["label"]].add(str(r.get("ad_id", "")))
            if r["label"] == "damaged":
                out["damage_boxes"] += len(boxes_from_row(r))
    out["total"] = len(rows)
    out["ads_total"] = len(set().union(*ads.values()))
    for label in LABELS:
        out[f"{label}_ads"] = len(ads[label])
    out["positive_ads"] = len(ads["damaged"] | ads["wreck"])
    audit_rows = [r for r in rows if r.get("dataset_split") == "audit"]
    out["audit_frames"] = len(audit_rows)
    out["audit_ads"] = len({str(r.get("ad_id", "")) for r in audit_rows})
    return out


def main():
    if "--stats" in sys.argv:
        s = stats()
        print(f"Размечено кадров: {s['total']}")
        for k, desc in LABELS.items():
            print(f"  {k:9} {s[k]:4} кадров, {s[f'{k}_ads']:3} объявлений   {desc}")
        need = 200 - s["positive_ads"]
        print(f"\nНезависимых объявлений damaged/wreck: {s['positive_ads']}. "
              "Это главный размер выборки для grouped CV.")
        print(f"Рамок локальных повреждений: {s['damage_boxes']}")
        print(f"Новый случайный audit holdout: {s['audit_ads']} объявлений, "
              f"{s['audit_frames']} кадров (legacy-метки туда не переносятся).")
        print("Ориентир для устойчивого локального замера — около 200: "
              f"{'хватает' if need <= 0 else f'ещё {need} объявлений'}")
        for k in ("parts", "wreck"):
            if s[k]:
                print(f"{k}: {s[k]} — отдельный класс, "
                      f"на них учим отдельно или объединяем позже")
        return

    q = queue()
    print(f"В очереди кадров: {len(q)}   (помеченных объявлений: "
          f"{int(q.suspect.sum())})")
    print(f"Уже размечено: {stats()['total']}")
    print("\nОткрыть разметку:  python -m kz.web  →  http://127.0.0.1:8000/damage")


if __name__ == "__main__":
    main()
