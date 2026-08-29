# -*- coding: utf-8 -*-
"""Экспорт ручных damage-меток в detector-ready COCO JSON.

Журнал ``data/photo_labels.csv`` остаётся источником истины и не меняется.
Экспорт — производный артефакт: его можно удалить и пересобрать. В train
попадают ``damaged`` (с рамкой) и ``intact`` (негатив без рамок); ``wreck``,
``parts`` и ``unclear`` намеренно исключены, потому что это другие визуальные
задачи. Новые ``audit``-строки экспортируются отдельно и никогда не
подмешиваются в train.

Запуск: python -m kz.ml.photo_dataset
Выход:  data/eda/photo_damage_train.coco.json
        data/eda/photo_damage_audit.coco.json
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageOps

from kz.report.photo_labels import read_journal

OUT_DIR = Path("data/eda")
DETECTION_LABELS = {"damaged", "intact"}
CATEGORIES = [{"id": 1, "name": "damage", "supercategory": "vehicle"}]


def _split(row: dict) -> str:
    # Пустое поле означает старую метку, которая уже участвовала в обучении.
    return str(row.get("dataset_split") or "train")


def _box(row: dict) -> tuple[float, float, float, float] | None:
    if not row.get("x1"):
        return None
    values = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"некорректная рамка {values} у {row.get('path')}")
    return values


def build_coco(rows: list[dict], split: str) -> dict:
    """Собрать COCO-словарь без копирования и изменения изображений."""
    if split not in {"train", "audit"}:
        raise ValueError(f"неизвестный split: {split!r}")

    latest = {}
    for row in rows:
        key = (str(row.get("ad_id", "")), str(row.get("position", "")))
        latest[key] = row

    selected = [r for r in latest.values()
                if r.get("label") in DETECTION_LABELS and _split(r) == split]
    selected.sort(key=lambda r: (str(r.get("ad_id", "")),
                                 int(r.get("position") or 0)))

    images, annotations = [], []
    annotation_id = 1
    for image_id, row in enumerate(selected, 1):
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"нет размеченного изображения: {path}")
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw)
            width, height = img.size

        images.append({
            "id": image_id,
            "file_name": str(path),
            "width": width,
            "height": height,
            "kz_ad_id": str(row.get("ad_id", "")),
            "position": int(row.get("position") or 0),
            "selection_source": str(row.get("selection_source") or "legacy"),
        })

        box = _box(row)
        if row.get("label") == "damaged" and box is None:
            raise ValueError(f"damaged без рамки: {path}")
        if row.get("label") != "damaged" or box is None:
            continue
        x1, y1, x2, y2 = box
        x, y = x1 * width, y1 * height
        w, h = (x2 - x1) * width, (y2 - y1) * height
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            # Округление убирает артефакты двоичной арифметики вроде
            # 49.99999999999999 из переносимого JSON.
            "bbox": [round(v, 6) for v in (x, y, w, h)],
            "area": round(w * h, 6),
            "iscrowd": 0,
        })
        annotation_id += 1

    return {
        "info": {
            "description": "KZ Market manually labelled vehicle damage",
            "split": split,
            "note": "Legacy labels are train; audit contains only newly sampled ads.",
        },
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }


def export(split: str, out: Path | None = None) -> Path:
    """Экспортировать split атомарной подменой файла."""
    _, rows = read_journal()
    payload = build_coco(rows, split)
    out = out or OUT_DIR / f"photo_damage_{split}.coco.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out)
    return out


def main() -> None:
    for split in ("train", "audit"):
        out = export(split)
        payload = json.loads(out.read_text(encoding="utf-8"))
        print(f"{split:5}: {len(payload['images'])} изображений, "
              f"{len(payload['annotations'])} рамок → {out}")


if __name__ == "__main__":
    main()
