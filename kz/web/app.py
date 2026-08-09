# -*- coding: utf-8 -*-
"""Веб-интерфейс: оценка цены и разметка вердиктов.

Тонкая обёртка над kz.web.service и kz.report.label_cards — своей логики
здесь почти нет, только маршруты и HTML. Так сделано, чтобы поведение можно
было проверить тестами без поднятия сервера.

Две страницы, по двум разным задачам:
  /estimate   продавец описывает машину и получает оценку, диапазон,
              разбор «почему столько», позицию среди похожих и замечания
              к объявлению;
  /label      разметка вердиктов — та же страница карточек, что и в
              label_cards --serve, только внутри общего приложения.

Запуск:  python -m kz.web
         → http://127.0.0.1:8000
"""

from __future__ import annotations

import html as _html

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from kz.report import label_cards
from kz.web import pages
from kz.web.service import full_estimate

app = FastAPI(title="KZ Car Market", docs_url="/api/docs")

# Карточки разметки собираются один раз при старте: запрос к базе тяжёлый,
# а список подозрительных меняется только после пересборки clean-слоя.
_cards_html: str | None = None
_cards_facts: dict = {}


def _cards():
    global _cards_html, _cards_facts
    if _cards_html is None:
        # Берём ПОЛНУЮ очередь разметки, а не только помеченных детектором.
        # Без контрольного слоя считается лишь precision: чтобы узнать, сколько
        # обмана детектор пропустил, надо проверять и те объявления, которые он
        # не трогал. Веб-интерфейс расходился здесь с консольным режимом
        # label_cards --serve --all, что тихо меняло смысл разметки.
        rows = label_cards.load_rows(include_queue=True)
        _cards_html = label_cards.build(rows, serve_mode=True)
        _cards_facts = label_cards.journal_facts(rows)
    return _cards_html


@app.get("/", response_class=HTMLResponse)
def index():
    return pages.index_page()


@app.get("/estimate", response_class=HTMLResponse)
def estimate_form():
    return pages.estimate_page()


@app.post("/api/estimate")
async def api_estimate(request: Request):
    """Оценка по характеристикам машины."""
    data = await request.json()
    from kz.ml.train_price_model import CAT_FEATURES, NUM_FEATURES

    car = {k: data.get(k) for k in CAT_FEATURES}
    # Все числовые поля приводим к числу здесь, на границе. Из формы они
    # приходят строками, и «8» < 5 роняет проверку объявления с невнятным
    # «'<' not supported between instances of 'str' and 'int'». Перечислять
    # поля руками нельзя: список признаков растёт, а забытое поле снова
    # приедет строкой.
    for k in list(NUM_FEATURES) + ["year"]:
        v = data.get(k)
        if v not in (None, ""):
            try:
                car[k] = float(v)
            except (TypeError, ValueError):
                return JSONResponse({"error": f"поле {k}: ожидается число, "
                                              f"получено {v!r}"}, status_code=400)
    if "year" in car and "age" not in car:
        from datetime import date
        car["age"] = date.today().year - int(car.pop("year")) + 1
    try:
        result = full_estimate(
            car,
            asking_price=float(data["asking_price"]) if data.get("asking_price") else None,
            text=str(data.get("text") or ""))
    except Exception as e:                      # noqa: BLE001 — ответ клиенту
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse(result)


@app.get("/photos/{path:path}")
def photo(path: str):
    """Локально скачанная фотография.

    Отдаём с диска, а не ссылкой на CDN: один из хостов kolesa уже отключён,
    и для 39% карточек внешние ссылки ведут в никуда.
    """
    from fastapi.responses import FileResponse
    from kz.collect.photo_fetch import PHOTO_DIR

    target = (PHOTO_DIR / path).resolve()
    # защита от выхода за каталог через ../
    if PHOTO_DIR.resolve() not in target.parents or not target.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target, media_type="image/jpeg")


@app.get("/label", response_class=HTMLResponse)
def label_page():
    return _cards()


@app.post("/verdict")
async def save_verdict(request: Request):
    """Вердикт разметчика. Путь совпадает с автономным режимом
    label_cards --serve, поэтому один и тот же JS работает в обоих."""
    _cards()                                    # гарантируем, что facts заполнены
    data = await request.json()
    ad_id = str(data.get("ad_id", ""))
    try:
        if ad_id not in _cards_facts:
            raise ValueError(f"неизвестный ad_id: {ad_id!r}")
        label_cards.upsert_verdict(ad_id, str(data.get("verdict", "")),
                                   str(data.get("comment", "")),
                                   _cards_facts[ad_id])
    except Exception as e:                      # noqa: BLE001
        return JSONResponse({"error": _html.escape(str(e))}, status_code=400)
    return JSONResponse({"ok": True})


@app.get("/api/health")
def health():
    from kz.web.service import get_model
    _, meta = get_model()
    return {
        "ok": True,
        "model_created": meta.get("created_at_utc"),
        "training_rows": meta.get("training_rows"),
    }
