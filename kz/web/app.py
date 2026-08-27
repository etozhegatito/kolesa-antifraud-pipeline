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
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from kz.report import label_cards
from kz.web import pages
from kz.web.service import full_estimate

# Публичный режим для контейнера, выложенного наружу. Оценка цены — вещь
# безобидная, а вот разметка вердиктов пишет в data/manual_labels.csv, то есть
# в размеченный вручную ground truth, на котором меряется весь антифрод.
# Открыть её анониму — значит дать любому желающему испортить единственный
# источник правды в проекте. Поэтому в публичном режиме /label и /verdict
# просто не существуют.
PUBLIC_DEMO = os.getenv("KZ_PUBLIC_DEMO", "").lower() in ("1", "true", "yes")

# Лимит запросов на адрес в публичном режиме. Число взято с большим запасом
# относительно живого человека: заполнить форму и нажать «оценить» чаще
# тридцати раз в минуту вручную нельзя. Цель не защита от злого умысла —
# от неё одним счётчиком не спастись, — а чтобы случайный цикл в чужом
# скрипте не съел весь процессор бесплатной машины.
RATE_LIMIT_PER_MIN = 30
RATE_WINDOW_SEC = 60

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


if PUBLIC_DEMO:
    import time
    from collections import defaultdict, deque

    _hits: dict[str, deque] = defaultdict(deque)

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        """Скользящее окно на адрес: храним времена запросов за последнюю
        минуту и отбрасываем всё, что старше.

        Счётчик живёт в памяти процесса, поэтому при нескольких машинах
        лимит становится «столько-то на машину». Для витрины этого хватает,
        а честный распределённый лимит потребовал бы Redis — лишняя
        зависимость ради задачи, которой пока нет.
        """
        ip = request.client.host if request.client else "?"
        now = time.monotonic()
        q = _hits[ip]
        while q and now - q[0] > RATE_WINDOW_SEC:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return JSONResponse({"error": "слишком часто, подождите минуту"},
                                status_code=429)
        q.append(now)
        return await call_next(request)


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


_damage_queue: list | None = None


def _damage_rows():
    """Очередь разметки повреждений, собирается один раз на процесс.

    Тяжёлый шаг — чтение векторов CLIP и запрос к базе; список меняется
    только после новой разметки, а её мы и так дописываем в журнал.
    """
    global _damage_queue
    if _damage_queue is None:
        from kz.report.photo_labels import queue
        q = queue()
        _damage_queue = [
            {"ad_id": str(r.ad_id), "position": int(r.position),
             "path": str(r.path), "suspect": bool(r.suspect),
             "price": (f"{r.price_tenge/1e6:.1f} млн"
                       if pd_notna(r.price_tenge) else "")}
            for r in q.itertuples()
        ]
    return _damage_queue


def pd_notna(v) -> bool:
    import pandas as pd
    return bool(pd.notna(v))


@app.get("/damage", response_class=HTMLResponse)
def damage_page():
    """Разметка повреждений рамками — тот же запрет, что и на /label.

    Пишет в data/photo_labels.csv, то есть в ручной труд, который не
    восстановить пересчётом. Наружу такое не открывается.
    """
    if PUBLIC_DEMO:
        return HTMLResponse("Разметка доступна только в локальном режиме.",
                            status_code=404)
    from kz.report.photo_labels import stats
    from kz.web.damage_page import page

    return page(_damage_rows(), stats())


@app.post("/damage/label")
async def damage_label(request: Request):
    """Сохранить метку кадра. Валидация на сервере, а не в браузере:
    страница может прислать что угодно, а журнал портить нельзя."""
    if PUBLIC_DEMO:
        return JSONResponse({"error": "not found"}, status_code=404)
    from kz.report.photo_labels import save_label, stats

    data = await request.json()
    known = {(r["ad_id"], r["position"]) for r in _damage_rows()}
    try:
        key = (str(data.get("ad_id")), int(data.get("position", -1)))
        if key not in known:
            raise ValueError(f"кадр не из очереди: {key}")
        save_label(str(data["ad_id"]), int(data["position"]),
                   str(data["path"]), str(data.get("label", "")),
                   box=data.get("box"), comment=str(data.get("comment") or ""))
    except Exception as e:                      # noqa: BLE001 — ответ клиенту
        return JSONResponse({"error": _html.escape(str(e))}, status_code=400)
    return JSONResponse({"ok": True, "stats": stats()})


@app.get("/label", response_class=HTMLResponse)
def label_page():
    if PUBLIC_DEMO:
        return HTMLResponse("Разметка доступна только в локальном режиме.",
                            status_code=404)
    return _cards()


@app.post("/verdict")
async def save_verdict(request: Request):
    """Вердикт разметчика. Путь совпадает с автономным режимом
    label_cards --serve, поэтому один и тот же JS работает в обоих."""
    if PUBLIC_DEMO:
        return JSONResponse({"error": "not found"}, status_code=404)
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
    val = meta.get("validation", {}).get("grouped_cv", {}).get("model", {})
    return {
        "ok": True,
        "model_created": meta.get("created_at_utc"),
        "training_rows": meta.get("training_rows"),
        "model_mape_pct": val.get("mape_pct"),
        "public_demo": PUBLIC_DEMO,
    }
