# -*- coding: utf-8 -*-
"""label_cards.py — офлайн-карточки для ручной разметки вердиктов.

ЗАЧЕМ. Чтобы поставить вердикт, надо ВИДЕТЬ машину. Раньше это значило
открывать kolesa.kz руками, и тут две проблемы:
  1) У архивных/удалённых объявлений страницы больше нет — посмотреть
     нечего, вердикт поставить нельзя.
  2) Ручной браузинг бьёт по тому же IP, что и джобы, и в бюджет
     catch_up НЕ попадает. Именно смесь «джобы + ручной браузинг» и
     положила IP 2026-07-23.

РЕШЕНИЕ. Фото лежат на CDN kcdn.kz — это ДРУГОЙ хост, и они переживают
смерть страницы (проверено: у archived и даже deleted объявлений фото
отдаются с HTTP 200). Всё остальное (весь текст, цена, avgPrice, бейдж,
цвет, пробег, damage-слова) у нас УЖЕ сохранено в базе. Значит карточку
можно собрать локально и разметить, ни разу не сходив на kolesa.kz.

Открытие получившегося HTML делает НОЛЬ запросов к kolesa.kz — только
подгрузку картинок с CDN. Бюджет kolesa не тратится вообще.

Запуск:  python -m kz.report.label_cards            → data/eda/label_cards.html
         python -m kz.report.label_cards --serve    → то же + локальный сервер, который
                                            ДОПИСЫВАЕТ вердикты в журнал сразу
                                            при нажатии (рекомендуемый режим)
         python -m kz.report.label_cards --all      → включить и residual-кандидатов
                                            из labeling_queue.csv, не только
                                            правиловых подозрительных

КАК СОХРАНЯЮТСЯ ВЕРДИКТЫ (три уровня, каждый со своей задачей):
  1) localStorage браузера — мгновенно, переживает перезагрузку и закрытие
     вкладки. Работает всегда, даже при открытии файла напрямую.
  2) data/manual_labels.csv — источник истины, читается clean.py. Пишется
     ТОЛЬКО в режиме --serve: страница, открытая как file://, писать на
     диск физически не может (ограничение браузера, не наша лень).
  3) корзина с копированием — запасной путь для файлового режима.

ОДНА СТРОКА НА ОБЪЯВЛЕНИЕ. Передумал — жми снова, и строка ОБНОВИТСЯ на
месте, а не продублируется. Смысл правила «журнал не терять» соблюдён:
перед первой правкой запуска рядом сохраняется предыдущая версия
(manual_labels.prev.csv), сама запись атомарна, вердикты не пропадают.
Накопленные ранее дубликаты сворачиваются: python -m kz.report.label_cards --dedupe
"""

import pathlib as _p
_expected = "label_cards.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import csv
import html
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kz.core.db import get_engine

OUT_HTML    = "data/eda/label_cards.html"
QUEUE_CSV   = "data/eda/labeling_queue.csv"
LABELS_CSV  = "data/manual_labels.csv"
# Состояние журнала до правок текущего запуска — точка восстановления.
# Файл один и перезаписывается, чтобы не разводить гору бэкапов.
LABELS_PREV = "data/manual_labels.prev.csv"

# Подсказки «как решать» по каждому флагу. Ключевой принцип проекта:
# fraud = ОБМАН, а не «плохая машина». Честно проданный хлам = legit.
FLAG_HELP = {
    "price_anomaly_low": (
        "Цена сильно ниже рынка для этой модели/года.",
        "fraud — если причина НЕ раскрыта: текст молчит о проблемах, фото "
        "целые, бейджа «Аварийная» нет. Это приманка «позвоните, а машины нет».",
        "legit — если дешевизна ОБЪЯСНЕНА: текст/фото/бейдж честно показывают "
        "аварию, гниль, отсутствие двигателя; либо это взнос по рассрочке "
        "(смотри «цена в месяц»); либо машина реально старая/убитая."),
    "young_car_cheap": (
        "Свежая машина по цене старой.",
        "fraud — если ничего не объясняет цену (целые фото, текст без проблем).",
        "legit — если это честно битая машина: бейдж «Аварийная/Не на ходу», "
        "фото с повреждениями, damage-слова в тексте."),
    "possible_repost": (
        "Похоже на дубль: то же авто выложено ещё раз.",
        "fraud — если на одну машину РАЗНЫЕ цены/пробеги/год: манипуляция "
        "выдачей, накрутка охвата, ценовая наживка.",
        "legit — если это просто перезалив (дилер поднял объявление): "
        "совпадают цвет, пробег, цена. Дубль ≠ обман."),
    "shared_photo_diff_car": (
        "Одно и то же фото у объявлений с разными атрибутами.",
        "fraud — если на фото ДРУГАЯ машина (украли чужое фото).",
        "legit — если это студийное/дилерское фото-шаблон или одна и та же "
        "машина в двух объявлениях. Смотри глазами: pHash не видит кропы."),
    "used_but_zero_mileage": (
        "У б/у машины пробег 0.",
        "fraud — только если явно врут про состояние («новая», «0 км» на "
        "убитой машине).",
        "legit / unknown — обычно это просто незаполненное поле, "
        "качество данных, а не обман."),
    "cheap_and_urgent": (
        "Дёшево + «срочно» в тексте.",
        "fraud — если срочность + необъяснимая дешевизна = давление на "
        "покупателя при отсутствии реальной машины.",
        "legit — если человек честно объясняет, почему торопится."),
}


def load_rows(include_queue: bool = False) -> pd.DataFrame:
    """Подозрительные из clean_data (+ опционально residual-кандидаты)."""
    eng = get_engine()
    cd = pd.read_sql("SELECT * FROM clean_data", eng, dtype={"ad_id": str})
    ids = set(cd.loc[cd["is_suspicious"] == 1, "ad_id"])
    stratum = {}
    if include_queue and Path(QUEUE_CSV).exists():
        q = pd.read_csv(QUEUE_CSV, dtype={"ad_id": str})
        ids |= set(q["ad_id"])
        stratum = dict(zip(q["ad_id"], q["sampling_stratum"]))
    rows = cd[cd["ad_id"].isin(ids)].copy()
    # Из какого слоя очереди объявление. Без этого разметчик не понимает,
    # ЧТО именно проверяет: у правиловых вопрос «верен ли флаг», а у
    # контрольных — «не пропустили ли мы обман», и это разные задачи.
    default = pd.Series(np.where(rows["is_suspicious"] == 1, "rule_positive", ""),
                        index=rows.index)
    rows["stratum"] = rows["ad_id"].map(stratum).fillna(default)

    # Доп. поля со страницы, которых нет в clean_data.
    enr = pd.read_sql("SELECT ad_id, options_text, page_condition, has_vin, "
                      "fetched_at FROM enriched", eng, dtype={"ad_id": str})
    rows = rows.merge(enr, on="ad_id", how="left")

    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", eng,
                         dtype={"ad_id": str})
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    photos = photos.sort_values(["ad_id", "position"])
    gal = photos.groupby("ad_id")["url"].apply(list)
    pos = photos.groupby("ad_id")["position"].apply(list)
    rows["photos"] = rows["ad_id"].map(gal)
    rows["photos"] = rows["photos"].apply(lambda v: v if isinstance(v, list) else [])
    # Позиции нужны, чтобы найти локально скачанный файл: он назван по
    # ad_id и позиции, а не по URL.
    rows["photo_positions"] = rows["ad_id"].map(pos)
    rows["photo_positions"] = rows["photo_positions"].apply(
        lambda v: v if isinstance(v, list) else [])

    # Уже размеченные помечаем, но НЕ выкидываем: удобно перепроверить.
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        done = lab[lab["verdict"].isin(["fraud", "legit"])]
        rows["existing_verdict"] = rows["ad_id"].map(
            dict(zip(done["ad_id"], done["verdict"])))
    else:
        rows["existing_verdict"] = None
    return rows.sort_values(["existing_verdict", "price_z"], na_position="first")


DEAD_HOSTS = {"alakt-photos-kl.kcdn.kz"}   # выведен из эксплуатации ~август 2026


def photo_src(ad_id: str, position: int, url: str, serve_mode: bool) -> str | None:
    """Откуда браузеру брать картинку.

    Приоритет у локальной копии: она грузится мгновенно и не зависит от того,
    жив ли сервер kolesa. Один из двух хостов раздачи уже исчез, и для 39%
    карточек ссылки ведут в никуда — там вернём None, чтобы карточка честно
    сказала «фото недоступны», а не показывала молча пустые рамки.
    """
    from kz.collect.photo_fetch import local_path

    p = local_path(ad_id, position)
    if p.exists():
        # в режиме сервера — через маршрут, в файловом — путь относительно
        # data/eda/, где лежит сама страница
        return f"/photos/{p.relative_to('data/photos')}" if serve_mode \
            else f"../photos/{p.relative_to('data/photos')}"
    host = url.split("/")[2] if "//" in url else ""
    return None if host in DEAD_HOSTS else url


def money(v) -> str:
    """Цена в читаемом виде; пусто — прочерк.

    Миллионы — только начиная с миллиона: «0.24М ₸» для 240 000 читается
    хуже, чем «240 000 ₸», а дешёвых объявлений среди подозрительных как
    раз много (приманки).
    """
    if pd.isna(v) or v is None:
        return "—"
    v = float(v)
    if v >= 1e6:
        return f"{v/1e6:.2f}".rstrip("0").rstrip(".") + "М ₸"
    return f"{int(v):,}".replace(",", " ") + " ₸"


def fmt(v) -> str:
    """Значение для таблички: NaN/None/пустое → прочерк, иначе экранируем."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    if isinstance(v, float) and float(v).is_integer():
        v = int(v)
    return html.escape(str(v))


# Градации отношения «цена объявления / средняя цена kolesa». Раньше было три
# грубых полосы, и на границе получалась бессмыслица вида «60% от среднего —
# цена в норме». Полосы уже, формулировки не спорят с процентом.
PRICE_BANDS = [
    (0.60, "сильно дешевле рынка"),
    (0.85, "заметно ниже среднего"),
    (1.15, "в пределах среднего"),
    (1.40, "выше среднего"),
    (float("inf"), "существенно выше рынка — приманкой быть не может"),
]


def price_band(ratio: float) -> str:
    """Словесная оценка отношения цены к средней по модели."""
    for limit, label in PRICE_BANDS:
        if ratio < limit:
            return label
    return PRICE_BANDS[-1][1]


def price_verdict_hint(row) -> str:
    """Кросс-чек с avgPrice самого kolesa — валидатор, НЕ признак модели."""
    avg = row.get("kolesa_avg_price")
    price = row.get("price_tenge")
    if pd.isna(avg) or avg is None or float(avg) <= 0 or pd.isna(price):
        return ""
    ratio = float(price) / float(avg)
    return (f"<b>{ratio*100:.0f}% от средней цены kolesa</b> ({money(avg)}) — "
            f"{price_band(ratio)}")


def card_html(row, idx: int, serve_mode: bool = False) -> str:
    """Одна карточка объявления.

    Компоновка подчинена задаче: главное — крупное фото (по миниатюре
    состояние машины не оценить), рядом факты, ниже текст. Подсказки
    «как решать» свёрнуты — они нужны на первых карточках, потом мешают.
    """
    aid = html.escape(str(row["ad_id"]))
    reasons = [r for r in str(row.get("suspicion_reasons") or "").split(";") if r]
    reasons = [p for r in reasons for p in r.split("|")]
    dead = row.get("status") in ("archived", "deleted")
    badge = row.get("page_status_badge")
    has_badge = badge not in (None, "-", "") and not pd.isna(badge)

    raw_photos = row["photos"]
    positions = row.get("photo_positions") or list(range(1, len(raw_photos) + 1))
    pairs = [(photo_src(str(row["ad_id"]), pos, u, serve_mode), u)
             for pos, u in zip(positions, raw_photos)]
    photos = [src for src, _ in pairs if src]
    n_dead = sum(1 for src, _ in pairs if src is None)
    if photos:
        thumbs = "".join(
            f'<button class="thumb{" on" if i == 0 else ""}" data-i="{i}" '
            f'aria-label="фото {i+1}">'
            f'<img loading="lazy" src="{html.escape(u)}" alt=""></button>'
            for i, u in enumerate(photos))
        gallery = (
            f'<div class="gal" data-photos=\'{html.escape(json.dumps(photos))}\'>'
            f'  <div class="hero"><img src="{html.escape(photos[0])}" alt="фото 1">'
            f'    <span class="zoom">нажми, чтобы увеличить</span>'
            f'    <span class="counter"><b>1</b>/{len(photos)}</span></div>'
            f'  <div class="thumbs">{thumbs}</div>'
            f'</div>')
    elif n_dead:
        gallery = ('<div class="gal"><div class="empty">Фотографии недоступны: '
                   f'сервер kolesa, где они лежали, отключён (было {n_dead} шт.). '
                   'Скачать их уже нельзя — решай по тексту и цифрам.</div></div>')
    else:
        gallery = '<div class="gal"><div class="empty">фото-URL не сохранены</div></div>'

    help_blocks = ""
    for r in reasons:
        if r in FLAG_HELP:
            what, fr, lg = FLAG_HELP[r]
            help_blocks += (
                f'<div class="help"><code>{html.escape(r)}</code>'
                f'<p class="hwhat">{html.escape(what)}</p>'
                f'<p class="hfraud"><span>fraud</span>{fr[7:] if fr.startswith("fraud —") else fr}</p>'
                f'<p class="hlegit"><span>legit</span>'
                f'{lg.split("—", 1)[1] if "—" in lg else lg}</p></div>')
    helps = (f'<details class="helps"><summary>как решать по этим флагам</summary>'
             f'{help_blocks}</details>' if help_blocks else "")

    facts = [
        ("Цена", money(row.get("price_tenge")), "big"),
        ("Год", fmt(row.get("year")), ""),
        ("Пробег, листинг", fmt(row.get("mileage_km")), ""),
        ("Пробег, страница", fmt(row.get("page_mileage_km")), ""),
        ("Двигатель", f'{fmt(row.get("engine_volume"))} · {fmt(row.get("engine_type"))}', ""),
        ("Коробка", fmt(row.get("transmission")), ""),
        ("Кузов", fmt(row.get("body_type")), ""),
        ("Цвет", fmt(row.get("color")), ""),
        ("Привод · руль", f'{fmt(row.get("drive"))} · {fmt(row.get("steering"))}', ""),
        ("Растаможен", fmt(row.get("customs_cleared")), ""),
        ("Состояние", fmt(row.get("page_condition")), ""),
        ("VIN указан", fmt(row.get("has_vin")), ""),
        ("price_z", fmt(round(float(row["price_z"]), 2)
                        if pd.notna(row.get("price_z")) else None), ""),
        ("Просмотров", fmt(row.get("views_count")), ""),
        ("Размещено", fmt(row.get("posted_date")), ""),
    ]
    facts_html = "".join(
        f'<div class="f {cls}"><dt>{k}</dt><dd>{v}</dd></div>' for k, v, cls in facts)

    texts = ""
    for label, key in [("Описание из листинга", "description"),
                       ("Комментарий продавца", "seller_comment"),
                       ("Опции", "options_text")]:
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
            texts += (f'<div class="txt"><div class="tl">{label}</div>'
                      f'<div class="tb">{html.escape(str(v))}</div></div>')
    if not texts:
        texts = ('<div class="empty">текста нет вообще — решай по фото и цене</div>')

    notes = ""
    dmg = row.get("damage_keywords")
    if dmg and not (isinstance(dmg, float) and pd.isna(dmg)) and str(dmg).strip():
        notes += (f'<div class="note legit-note"><b>{html.escape(str(dmg))}</b>'
                  f'<span>продавец сам раскрыл проблему — в пользу legit</span></div>')
    if has_badge:
        notes += (f'<div class="note legit-note"><b>бейдж: '
                  f'{html.escape(str(badge))}</b><span>сайт сам помечает машину '
                  f'проблемной, дешевизна объяснена — в пользу legit</span></div>')

    hint = price_verdict_hint(row)
    if hint:
        notes += f'<div class="note price-note">{hint}</div>'

    STRATUM_HELP = {
        "rule_positive": ("правила пометили",
                          "Вопрос: флаг верный? Обман — или объяснимая дешевизна."),
        "residual_candidate": ("модель: подозрительно дёшево",
                               "Правила молчат, но цена ниже ожидаемой для такой "
                               "машины. Вопрос тот же: обман или объяснимо."),
        "random_control": ("контрольное, детектор НЕ помечал",
                           "Почти наверняка legit — и это нормальный, ожидаемый "
                           "ответ. Смысл проверки в другом: найти обман, который "
                           "детектор пропустил. Без этого нельзя посчитать "
                           "полноту (recall)."),
    }
    st = str(row.get("stratum") or "")
    stratum_html = ""
    if st in STRATUM_HELP:
        title, hint = STRATUM_HELP[st]
        cls = "s-control" if st == "random_control" else "s-flagged"
        stratum_html = (f'<div class="stratum {cls}"><b>{title}</b>'
                        f'<span>{hint}</span></div>')

    ev = row.get("existing_verdict")
    ev_html = (f'<div class="note done-note">уже размечено: '
               f'<b>{html.escape(str(ev))}</b><span>можно перепроверить</span></div>'
               if ev and not pd.isna(ev) else "")

    status_html = (
        f'<span class="dead">{html.escape(str(row.get("status")))} · страницы '
        f'больше нет, но фото живы</span>' if dead else
        f'<a class="live" href="https://kolesa.kz/a/show/{aid}" target="_blank" '
        f'rel="noreferrer">открыть на kolesa ↗<em>тратит лимит IP</em></a>')

    title = (f'{html.escape(str(row.get("brand") or ""))} '
             f'{html.escape(str(row.get("model") or ""))}').strip()

    return f"""
<article class="card" id="ad{aid}" data-id="{aid}" data-idx="{idx}" data-stratum="{st}">
  <header>
    <div class="ttl">
      <h2>{title} <span class="yr">{fmt(row.get("year"))}</span></h2>
      <div class="meta"><span class="num">#{idx + 1}</span> id {aid} · {status_html}</div>
    </div>
    <div class="flags">
      {"".join(f'<span class="flag">{html.escape(r)}</span>' for r in reasons)}
    </div>
  </header>
  {stratum_html}
  {ev_html}
  <div class="body">
    <div class="left">{gallery}</div>
    <div class="right">
      <dl class="facts">{facts_html}</dl>
      {notes}
    </div>
  </div>
  <div class="texts">{texts}</div>
  {helps}
  <footer class="actions">
    <button class="bfraud" data-v="fraud">fraud<kbd>F</kbd></button>
    <button class="blegit" data-v="legit">legit<kbd>L</kbd></button>
    <button class="bunk" data-v="unknown">unknown<kbd>U</kbd></button>
    <input class="cmt" placeholder="почему — попадёт в колонку comment">
    <span class="picked"></span>
  </footer>
</article>"""


def build(rows: pd.DataFrame, serve_mode: bool = False) -> str:
    cards = "".join(card_html(r, i, serve_mode)
                    for i, (_, r) in enumerate(rows.iterrows()))
    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_done = int(rows["existing_verdict"].notna().sum())
    n_nophoto = sum(1 for _, r in rows.iterrows()
                    if r["photos"] and not any(
                        photo_src(str(r["ad_id"]), p, u, serve_mode)
                        for p, u in zip(r.get("photo_positions") or [], r["photos"])))
    mode = ("вердикты пишутся в журнал" if serve_mode
            else "черновик в браузере — журнал не пишется")
    return (TEMPLATE
            .replace("__CARDS__", cards)
            .replace("__N__", str(len(rows)))
            .replace("__NDEAD__", str(n_dead))
            .replace("__NOPHOTO__", str(n_nophoto))
            .replace("__NDONE__", str(n_done))
            .replace("__SERVER__", "true" if serve_mode else "false")
            .replace("__MODECLS__", "live" if serve_mode else "draft")
            .replace("__MODE__", mode)
            .replace("__LABELS__", html.escape(LABELS_CSV)))


# Шаблон отдельно от Python-строк с фигурными скобками CSS: подстановка
# через .replace(), а не f-string — иначе пришлось бы экранировать всё CSS.
# Строка СЫРАЯ (r"""), чтобы \n внутри JS писались как в JS, без двойного
# экранирования.
TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карточки для разметки</title>
<style>
/* ── Тема. По умолчанию системная, кнопка ставит data-theme на <html>. ── */
:root{
  --bg:#0c0f16; --surface:#131824; --surface2:#1a2030; --line:#232b3d;
  --text:#e7eaf2; --muted:#8f98ab; --faint:#5f6878;
  --accent:#7aa7ff; --accent-bg:#14203a;
  --fraud:#ff8b8b; --fraud-bg:#2a1518; --fraud-line:#5e2a2e;
  --legit:#7fe0a5; --legit-bg:#102319; --legit-line:#265f3d;
  --warn:#ffc470; --warn-bg:#241c0e;
  --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px -12px rgba(0,0,0,.6);
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --surface:#ffffff; --surface2:#f2f4f7; --line:#e2e6ed;
  --text:#161a22; --muted:#5f6773; --faint:#98a1af;
  --accent:#2563c9; --accent-bg:#eaf1ff;
  --fraud:#b4232c; --fraud-bg:#fdeeef; --fraud-line:#f2c4c7;
  --legit:#1a7a48; --legit-bg:#eaf7ef; --legit-line:#bde3cb;
  --warn:#8a5a00; --warn-bg:#fdf3e0;
  --shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px -14px rgba(16,24,40,.18);
}
@media (prefers-color-scheme: light){
  :root:not([data-theme]){
    --bg:#f6f7f9; --surface:#ffffff; --surface2:#f2f4f7; --line:#e2e6ed;
    --text:#161a22; --muted:#5f6773; --faint:#98a1af;
    --accent:#2563c9; --accent-bg:#eaf1ff;
    --fraud:#b4232c; --fraud-bg:#fdeeef; --fraud-line:#f2c4c7;
    --legit:#1a7a48; --legit-bg:#eaf7ef; --legit-line:#bde3cb;
    --warn:#8a5a00; --warn-bg:#fdf3e0;
    --shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px -14px rgba(16,24,40,.18);
  }
}

/* ── Типографика. Системные шрифты: нативно, резко, без внешних загрузок
      (страница обязана работать офлайн). Цифры табличные, чтобы столбцы
      фактов и цены не «плясали». ── */
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",
    "Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,
    "Noto Sans",sans-serif;
  font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;
  font-variant-numeric:tabular-nums; font-kerning:normal;
}
h1,h2{font-weight:600; letter-spacing:-.015em; line-height:1.25; margin:0}
h1{font-size:1.5rem}
h2{font-size:1.2rem}
code,kbd,.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,
  Consolas,"Liberation Mono",monospace}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}

.wrap{max-width:1160px; margin:0 auto; padding:0 20px 80px}

/* ── Верхняя панель ── */
.top{position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 88%,transparent);
     backdrop-filter:saturate(1.4) blur(10px); border-bottom:1px solid var(--line);
     margin:0 -20px 24px; padding:0 20px}
.progress{height:2px; background:var(--line); margin:0 -20px}
.progress i{display:block; height:2px; width:0; background:var(--accent);
     transition:width .25s ease}
.topin{display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding:12px 0}
.topin h1{margin-right:auto}
.count{color:var(--muted); font-size:.875rem}
.count b{color:var(--text)}
.tbtn{background:var(--surface); color:var(--text); border:1px solid var(--line);
  border-radius:8px; padding:6px 12px; font:inherit; font-size:.875rem;
  cursor:pointer; line-height:1.4}
.tbtn:hover{background:var(--surface2)}
.tbtn.on{border-color:var(--accent); color:var(--accent); background:var(--accent-bg)}

.lede{color:var(--muted); font-size:.9375rem; margin:0 0 16px; max-width:70ch}
.safe{background:var(--legit-bg); border:1px solid var(--legit-line);
  border-radius:10px; padding:12px 14px; margin:0 0 18px; font-size:.9375rem;
  max-width:80ch}
.safe b{color:var(--legit)}
.keys{display:flex; gap:16px; flex-wrap:wrap; color:var(--muted);
  font-size:.8125rem; margin:0 0 22px}
kbd{background:var(--surface2); border:1px solid var(--line); border-bottom-width:2px;
  border-radius:5px; padding:1px 6px; font-size:.75rem; color:var(--text)}

/* ── Корзина вердиктов ── */
.basket{background:var(--surface); border:1px solid var(--line);
  border-radius:12px; margin:0 0 28px; box-shadow:var(--shadow)}
.basket summary{cursor:pointer; padding:13px 16px; font-weight:500;
  display:flex; align-items:center; gap:10px; list-style:none}
.basket summary::-webkit-details-marker{display:none}
.basket summary::before{content:"▸"; color:var(--muted); font-size:.8em}
.basket[open] summary::before{content:"▾"}
.basket .in{padding:0 16px 16px}
.basket textarea{width:100%; height:130px; resize:vertical; background:var(--bg);
  color:var(--text); border:1px solid var(--line); border-radius:8px; padding:11px 12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.8125rem;
  line-height:1.6}
.basket .row{display:flex; gap:9px; margin-top:10px; flex-wrap:wrap}
.hintline{color:var(--muted); font-size:.8125rem; margin:9px 0 0}

/* ── Карточка ── */
.card{background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:20px; margin:0 0 22px; scroll-margin-top:76px; box-shadow:var(--shadow);
  border-left:3px solid transparent}
.card.cur{border-color:var(--accent)}
.card[data-verdict="fraud"]{border-left-color:var(--fraud)}
.card[data-verdict="legit"]{border-left-color:var(--legit)}
.card[data-verdict="unknown"]{border-left-color:var(--faint)}
body.hide-done .card[data-verdict], body.hide-done .card.done{display:none}
body.only-control .card:not([data-stratum="random_control"]){display:none}

.card header{display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; flex-wrap:wrap; margin-bottom:14px}
.yr{color:var(--muted); font-weight:400}
.meta{color:var(--muted); font-size:.8125rem; margin-top:5px}
.num{color:var(--faint)}
.dead{color:var(--warn)}
.live em{color:var(--faint); font-style:normal; font-size:.9em; margin-left:6px}
.flags{display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end}
.flag{background:var(--fraud-bg); border:1px solid var(--fraud-line); color:var(--fraud);
  border-radius:999px; padding:2px 10px; font-size:.75rem; white-space:nowrap;
  font-family:ui-monospace,Menlo,monospace}

.body{display:grid; gap:20px; grid-template-columns:1fr}
@media (min-width:1000px){ .body{grid-template-columns:minmax(0,1.5fr) minmax(290px,1fr)} }

/* ── Галерея: крупное фото главное, миниатюры переключают ── */
/* Высота фиксированная, а не aspect-ratio: среди фото есть вертикальные, и
   в жёсткой рамке 4:3 они сжимались в узкую полоску — по такой картинке
   состояние машины не оценить. Фиксированная высота + contain даёт крупное
   изображение при любой ориентации и не дёргает вёрстку при перелистывании. */
.hero{position:relative; background:var(--surface2); border:1px solid var(--line);
  border-radius:11px; overflow:hidden; cursor:zoom-in;
  height:clamp(320px, 54vh, 580px)}
.hero img{width:100%; height:100%; object-fit:contain; display:block}
.hero .zoom,.hero .counter{position:absolute; bottom:9px; font-size:.75rem;
  background:rgba(8,10,15,.72); color:#f0f2f7; padding:3px 9px; border-radius:6px;
  backdrop-filter:blur(4px)}
.hero .zoom{left:9px; opacity:0; transition:opacity .15s}
.hero:hover .zoom{opacity:1}
.hero .counter{right:9px}
.thumbs{display:flex; gap:7px; margin-top:9px; overflow-x:auto; padding-bottom:4px;
  scrollbar-width:thin}
.thumb{flex:0 0 auto; width:74px; height:56px; padding:0; border-radius:7px;
  border:1px solid var(--line); background:var(--surface2); overflow:hidden;
  cursor:pointer; opacity:.62; transition:opacity .15s, border-color .15s}
.thumb img{width:100%; height:100%; object-fit:cover; display:block}
.thumb:hover{opacity:.9}
.thumb.on{opacity:1; border-color:var(--accent); box-shadow:0 0 0 1px var(--accent)}
.empty{color:var(--muted); font-style:italic; padding:14px 0; font-size:.9375rem}

/* ── Факты ── */
.facts{margin:0; display:grid; gap:0}
.f{display:flex; justify-content:space-between; align-items:baseline; gap:12px;
  padding:7px 0; border-bottom:1px solid var(--line); font-size:.875rem}
.f:last-child{border-bottom:none}
.f dt{color:var(--muted); margin:0}
.f dd{margin:0; text-align:right; font-weight:500}
.f.big{padding:4px 0 10px; border-bottom:1px solid var(--line)}
.f.big dt{font-size:.875rem}
.f.big dd{font-size:1.35rem; font-weight:600; letter-spacing:-.01em}

.note{border-radius:9px; padding:10px 12px; margin-top:11px; font-size:.875rem;
  line-height:1.5}
.note b{display:block; margin-bottom:2px}
.note span{color:var(--muted); display:block}
.legit-note{background:var(--legit-bg); border:1px solid var(--legit-line)}
.legit-note b{color:var(--legit)}
.price-note{background:var(--accent-bg); border:1px solid var(--line)}
.stratum{border-radius:9px;padding:10px 13px;margin-bottom:12px;font-size:.9rem}
.stratum b{display:block;margin-bottom:3px}
.stratum span{color:var(--muted);display:block;line-height:1.45}
.s-flagged{background:var(--fraud-bg);border:1px solid var(--fraud-line)}
.s-flagged b{color:var(--fraud)}
.s-control{background:var(--accent-bg);border:1px solid var(--line)}
.s-control b{color:var(--accent)}
.done-note{background:var(--surface2); border:1px solid var(--line); margin:0 0 14px}

/* ── Текст объявления: это читают внимательно, поэтому мера строки
      ограничена, а межстрочный интервал больше. ── */
.texts{margin-top:20px}
.txt+.txt{margin-top:14px}
.tl{color:var(--muted); font-size:.75rem; text-transform:uppercase;
  letter-spacing:.06em; font-weight:500; margin-bottom:5px}
.tb{background:var(--bg); border:1px solid var(--line); border-radius:9px;
  padding:12px 14px; white-space:pre-wrap; max-width:78ch; line-height:1.7}

/* ── Подсказки по флагам ── */
.helps{margin-top:18px; border-top:1px solid var(--line); padding-top:12px}
.helps summary{cursor:pointer; color:var(--muted); font-size:.875rem;
  list-style:none}
.helps summary::-webkit-details-marker{display:none}
.helps summary::before{content:"▸ "; font-size:.8em}
.helps[open] summary::before{content:"▾ "}
.help{margin:14px 0 0; padding-left:13px; border-left:2px solid var(--line)}
.help code{font-size:.75rem; color:var(--muted)}
.help p{margin:4px 0 0; font-size:.875rem; line-height:1.55}
.hwhat{color:var(--text)}
.hfraud span,.hlegit span{font-family:ui-monospace,Menlo,monospace; font-size:.75rem;
  padding:1px 7px; border-radius:5px; margin-right:7px}
.hfraud span{background:var(--fraud-bg); color:var(--fraud)}
.hlegit span{background:var(--legit-bg); color:var(--legit)}

/* ── Кнопки вердикта ── */
.actions{display:flex; gap:9px; align-items:center; flex-wrap:wrap; margin-top:18px;
  border-top:1px solid var(--line); padding-top:16px}
.actions button{display:inline-flex; align-items:center; gap:7px; font:inherit;
  font-size:.9375rem; font-weight:500; padding:8px 15px; border-radius:9px;
  cursor:pointer; border:1px solid var(--line); background:var(--surface2);
  color:var(--text); transition:transform .08s}
.actions button:active{transform:translateY(1px)}
.actions button kbd{border-bottom-width:1px; opacity:.65}
.bfraud{background:var(--fraud-bg); border-color:var(--fraud-line); color:var(--fraud)}
.blegit{background:var(--legit-bg); border-color:var(--legit-line); color:var(--legit)}
.cmt{flex:1 1 240px; min-width:200px; background:var(--bg); color:var(--text);
  border:1px solid var(--line); border-radius:9px; padding:9px 12px; font:inherit;
  font-size:.9375rem}
.cmt:focus{outline:2px solid var(--accent); outline-offset:-1px; border-color:transparent}
.picked{font-size:.875rem; font-weight:500; color:var(--muted)}
.picked[data-state="local"]{color:var(--warn)}
.picked[data-state="saved"]{color:var(--legit)}
.picked[data-state="error"]{color:var(--fraud)}
.mode{font-size:.75rem; padding:3px 10px; border-radius:999px; white-space:nowrap}
.mode.live{background:var(--legit-bg); border:1px solid var(--legit-line);
  color:var(--legit)}
.mode.draft{background:var(--warn-bg); border:1px solid var(--line);
  color:var(--warn)}

/* ── Лайтбокс ── */
#box{position:fixed; inset:0; z-index:100; display:none; align-items:center;
  justify-content:center; background:rgba(6,8,12,.94); backdrop-filter:blur(3px)}
#box.open{display:flex}
#box img{max-width:93vw; max-height:88vh; object-fit:contain; border-radius:8px}
#box .bx{position:absolute; background:rgba(255,255,255,.1); color:#fff;
  border:none; border-radius:9px; width:46px; height:60px; font-size:1.4rem;
  cursor:pointer}
#box .bx:hover{background:rgba(255,255,255,.2)}
#box .prev{left:18px} #box .next{right:18px}
#box .cls{top:18px; right:18px; height:40px; width:40px; font-size:1.1rem}
#box .n{position:absolute; bottom:20px; color:#cfd4de; font-size:.875rem}
</style>

<div class="wrap">
<div class="top">
  <div class="progress"><i id="bar"></i></div>
  <div class="topin">
    <h1>Разметка антифрода</h1>
    <span class="count"><b id="cnt">0</b> из __N__ размечено</span>
    <span class="count" id="restored"></span>
    <span class="mode __MODECLS__">__MODE__</span>
    <button class="tbtn" id="filter">скрыть размеченные</button>
    <button class="tbtn" id="only-control">только контрольные</button>
    <button class="tbtn" id="theme">тема</button>
  </div>
</div>

<p class="lede">__N__ объявлений. Уже есть вердикт: __NDONE__.
__NDEAD__ с закрытой страницей на kolesa, __NOPHOTO__ без доступных фотографий —
у них сервер, где лежали снимки, отключён.</p>

<div class="safe"><b>Страница не обращается к kolesa.kz.</b> Фотографии грузятся
с CDN (другой хост), текст и цифры взяты из локальной базы, поэтому разметка не
расходует суточный лимит запросов. Единственное исключение — ссылка «открыть на
kolesa» в карточке.</div>

<div class="keys">
  <span><kbd>J</kbd><kbd>K</kbd> следующая / предыдущая</span>
  <span><kbd>F</kbd> fraud &nbsp;<kbd>L</kbd> legit &nbsp;<kbd>U</kbd> unknown</span>
  <span><kbd>←</kbd><kbd>→</kbd> перелистнуть фото</span>
  <span><kbd>C</kbd> комментарий</span>
</div>

<details class="basket" open>
  <summary>Отмечено в этой сессии — <span id="cnt2">0</span></summary>
  <div class="in">
    <textarea id="out" readonly placeholder="Нажимай fraud / legit / unknown в карточках."></textarea>
    <div class="row">
      <button class="tbtn" id="copy">копировать всё</button>
      <button class="tbtn" id="clear">очистить черновик</button>
    </div>
    <p class="hintline" id="baskethint"></p>
  </div>
</details>

__CARDS__
</div>

<div id="box">
  <button class="bx cls" title="Esc">✕</button>
  <button class="bx prev" title="←">‹</button>
  <img id="boximg" alt="">
  <button class="bx next" title="→">›</button>
  <span class="n" id="boxn"></span>
</div>

<script>
const SERVER = __SERVER__;   /* true — запущено через --serve, можно писать в журнал */
const cards = Array.from(document.querySelectorAll('.card'));
/* Объявления, по которым вердикт уже лежит в журнале. */
const ALREADY = new Set(cards.filter(c => c.querySelector('.done-note'))
                             .map(c => c.dataset.id));
const picks = new Map();
let cur = 0;

/* Кавычки и переводы строк экранируются по правилам CSV, иначе комментарий
   с запятой сдвинул бы колонки журнала. */
function esc(s){ return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }

function render(){
  const lines = [];
  /* Схема журнала: ad_id, 7 описательных, verdict, comment, затем колонки
     слоя. Число запятых должно совпадать с заголовком, иначе вставленная
     строка сдвинет колонки. */
  for (const [id, v] of picks) lines.push(id + ',,,,,,,,' + v + ',,');
  document.getElementById('out').value = lines.join('\n');
  /* Счётчик показывает ИТОГ: что уже в журнале плюс отмеченное сейчас.
     Раньше он показывал только черновик, и при открытии на другом порту
     (у localStorage своя память на каждый адрес) выглядело так, будто
     разметка пропала. */
  const fresh = [...picks.keys()].filter(id => !ALREADY.has(id)).length;
  document.getElementById('cnt').textContent = ALREADY.size + fresh;
  document.getElementById('cnt2').textContent = picks.size;
  document.getElementById('bar').style.width =
    (cards.length ? picks.size / cards.length * 100 : 0) + '%';
}

/* Выбор живёт в трёх местах, и это не дублирование, а разные задачи:
   picks — текущая сессия (корзина/копирование);
   localStorage — переживает перезагрузку и закрытие вкладки;
   журнал на диске — единственный источник истины, пишется сервером. */
const STORE = 'label_cards_picks';

function saveLocal(){
  const obj = {};
  for (const [id, v] of picks) obj[id] = v;
  try { localStorage.setItem(STORE, JSON.stringify(obj)); } catch (e) {}
}

function loadLocal(){
  try { return JSON.parse(localStorage.getItem(STORE) || '{}'); }
  catch (e) { return {}; }
}

function mark(card, state, text){
  const el = card.querySelector('.picked');
  el.dataset.state = state;
  el.textContent = text;
}

function setVerdict(card, v){
  if (!card) return;
  const cmt = card.querySelector('.cmt').value.trim();
  picks.set(card.dataset.id, v + ',' + esc(cmt));
  card.dataset.verdict = v;
  mark(card, 'local', '→ ' + v);
  saveLocal();
  render();
  if (!SERVER) return;                    /* file:// — только копипаста */
  fetch('/verdict', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ad_id: card.dataset.id, verdict: v, comment: cmt}),
  }).then(r => r.json())
    .then(d => mark(card, d.ok ? 'saved' : 'error',
                    d.ok ? '✓ в журнале' : '✗ ' + (d.error || 'ошибка')))
    /* Сеть отвалилась — выбор всё равно в localStorage, не потеряется. */
    .catch(() => mark(card, 'error', '✗ не сохранено, есть в браузере'));
}

function focusCard(i){
  if (!cards.length) return;
  cur = Math.max(0, Math.min(cards.length - 1, i));
  cards.forEach((c, j) => c.classList.toggle('cur', j === cur));
  cards[cur].scrollIntoView({block: 'start', behavior: 'smooth'});
}

/* ── Галереи ── */
cards.forEach(card => {
  if (card.querySelector('.done-note')) card.classList.add('done');
  card.querySelectorAll('.actions button').forEach(b => {
    b.onclick = () => { focusCard(+card.dataset.idx); setVerdict(card, b.dataset.v); };
  });
  const gal = card.querySelector('.gal');
  if (!gal || !gal.dataset.photos) return;
  const photos = JSON.parse(gal.dataset.photos);
  const hero = gal.querySelector('.hero img');
  const counter = gal.querySelector('.counter b');
  let i = 0;
  const show = n => {
    i = (n + photos.length) % photos.length;
    hero.src = photos[i];
    counter.textContent = i + 1;
    gal.querySelectorAll('.thumb').forEach((t, j) => t.classList.toggle('on', j === i));
  };
  gal.querySelectorAll('.thumb').forEach(t => t.onclick = () => show(+t.dataset.i));
  gal.querySelector('.hero').onclick = () => openBox(photos, i);
  card._show = show;
  card._at = () => i;
});

/* ── Лайтбокс ── */
const box = document.getElementById('box');
let bp = [], bi = 0;
function paint(){
  document.getElementById('boximg').src = bp[bi];
  document.getElementById('boxn').textContent = (bi + 1) + ' / ' + bp.length;
}
function openBox(p, i){ bp = p; bi = i; box.classList.add('open'); paint(); }
function moveBox(d){ bi = (bi + d + bp.length) % bp.length; paint(); }
function closeBox(){ box.classList.remove('open'); }
box.querySelector('.cls').onclick = closeBox;
box.querySelector('.prev').onclick = e => { e.stopPropagation(); moveBox(-1); };
box.querySelector('.next').onclick = e => { e.stopPropagation(); moveBox(1); };
box.onclick = e => { if (e.target === box) closeBox(); };

/* ── Клавиатура ── */
document.addEventListener('keydown', e => {
  if (box.classList.contains('open')){
    if (e.key === 'Escape') closeBox();
    else if (e.key === 'ArrowRight') moveBox(1);
    else if (e.key === 'ArrowLeft') moveBox(-1);
    return;
  }
  if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName)){
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const card = cards[cur];
  const k = e.key.toLowerCase();
  if (k === 'j') focusCard(cur + 1);
  else if (k === 'k') focusCard(cur - 1);
  else if (k === 'f') setVerdict(card, 'fraud');
  else if (k === 'l') setVerdict(card, 'legit');
  else if (k === 'u') setVerdict(card, 'unknown');
  else if (k === 'c'){ card && card.querySelector('.cmt').focus(); e.preventDefault(); }
  else if (e.key === 'ArrowRight'){ card && card._show && card._show(card._at() + 1); }
  else if (e.key === 'ArrowLeft'){ card && card._show && card._show(card._at() - 1); }
  else return;
  if (k === 'j' || k === 'k') e.preventDefault();
});

/* ── Панель ── */
document.getElementById('copy').onclick = () => {
  const t = document.getElementById('out');
  if (!t.value) return;
  const text = t.value + '\n';
  /* На file:// clipboard API бывает недоступен — тогда старый способ. */
  const fallback = () => {
    t.removeAttribute('readonly'); t.select();
    document.execCommand('copy');
    t.setAttribute('readonly', ''); window.getSelection().removeAllRanges();
  };
  if (navigator.clipboard) navigator.clipboard.writeText(text).catch(fallback);
  else fallback();
  const b = document.getElementById('copy');
  b.textContent = 'скопировано';
  setTimeout(() => b.textContent = 'копировать всё', 1400);
};
document.getElementById('clear').onclick = () => {
  /* Чистит только черновик в браузере. Journal на диске не трогаем никогда —
     он append-only, и уже записанные вердикты остаются валидными. */
  if (!confirm('Очистить черновик в браузере? Уже записанные в журнал '
             + 'вердикты останутся — файл только дописывается.')) return;
  picks.clear();
  try { localStorage.removeItem(STORE); } catch (e) {}
  cards.forEach(c => {
    delete c.dataset.verdict;
    c.querySelector('.picked').textContent = '';
  });
  render();
};
document.getElementById('filter').onclick = e => {
  document.body.classList.toggle('hide-done');
  const on = document.body.classList.contains('hide-done');
  e.target.classList.toggle('on', on);
  e.target.textContent = on ? 'показать все' : 'скрыть размеченные';
};
document.getElementById('only-control').onclick = e => {
  /* Контрольные — обычные объявления, которых детектор НЕ помечал. Только по
     ним считается полнота: сколько обмана мы пропустили. Фильтр нужен, чтобы
     не искать их глазами среди помеченных. */
  document.body.classList.toggle('only-control');
  const on = document.body.classList.contains('only-control');
  e.target.classList.toggle('on', on);
  e.target.textContent = on ? 'показать все' : 'только контрольные';
  focusCard(0);
};
document.getElementById('theme').onclick = () => {
  const root = document.documentElement;
  const now = root.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  root.dataset.theme = now === 'dark' ? 'light' : 'dark';
};

/* Восстановление черновика: выборы прошлой сессии видны сразу, не надо
   вспоминать, где остановился. Комментарий тоже возвращаем в поле. */
(() => {
  const saved = loadLocal();
  let n = 0;
  for (const card of cards){
    const v = saved[card.dataset.id];
    if (!v) continue;
    picks.set(card.dataset.id, v);
    const i = v.indexOf(',');
    const verdict = i < 0 ? v : v.slice(0, i);
    let cmt = i < 0 ? '' : v.slice(i + 1);
    if (cmt.startsWith('"') && cmt.endsWith('"'))
      cmt = cmt.slice(1, -1).replace(/""/g, '"');
    card.dataset.verdict = verdict;
    card.querySelector('.cmt').value = cmt;
    mark(card, 'local', '→ ' + verdict);
    n++;
  }
  if (n) document.getElementById('restored').textContent =
    'восстановлено из браузера: ' + n;
})();

/* Подсказка зависит от режима: в серверном вердикты уже в журнале, и
   советовать копипасту значит путать. */
const _hint = document.getElementById('baskethint');
if (_hint) _hint.innerHTML = SERVER
  ? 'Вердикты уже записаны в <b>__LABELS__</b> — копировать ничего не нужно. '
    + 'Этот список только показывает, что ты отметил в текущей сессии. '
    + 'Дальше: <span class="mono">python -m kz.ops.run_all --ml</span>.'
  : 'Страница открыта как файл, поэтому писать в журнал не может. '
    + 'Скопируй строки и допиши в конец <b>__LABELS__</b>. '
    + 'Надёжнее запускать через <span class="mono">python -m kz.web</span>.';

focusCard(0);
render();
</script>
"""


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


def serve(html: str, facts: dict, port: int = 8765) -> None:
    """Локальный сервер: отдаёт карточки и дописывает вердикты в журнал.

    Нужен потому, что страница, открытая как file://, писать на диск не может
    в принципе — а без записи выборы приходилось переносить копипастой.
    Слушаем только 127.0.0.1: инструмент локальный, наружу его открывать
    незачем. Пишем строго через append_verdict (валидация + append-only).
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import unquote

    page = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, page, "text/html; charset=utf-8")
            if path.startswith("/photos/"):
                # Скачанные картинки. Путь нормализуем и проверяем, что он не
                # вылезает за каталог фотографий: иначе через ../ можно было бы
                # прочитать любой файл на диске.
                from kz.collect.photo_fetch import PHOTO_DIR
                rel = unquote(path[len("/photos/"):])
                target = (PHOTO_DIR / rel).resolve()
                if PHOTO_DIR.resolve() in target.parents and target.is_file():
                    return self._send(200, target.read_bytes(), "image/jpeg")
                return self._send(404, b"not found", "text/plain")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/verdict":
                return self._send(404, b'{"error":"not found"}',
                                  "application/json")
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n) or b"{}")
                ad_id = str(data.get("ad_id", ""))
                # ad_id принимаем только из числа показанных карточек:
                # запись в журнал не должна зависеть от того, что пришло в теле
                if ad_id not in facts:
                    raise ValueError(f"неизвестный ad_id: {ad_id!r}")
                upsert_verdict(ad_id, str(data.get("verdict", "")),
                               str(data.get("comment", "")), facts[ad_id])
            except Exception as e:                  # noqa: BLE001 — ответ клиенту
                return self._send(400, json.dumps({"error": str(e)},
                                  ensure_ascii=False).encode(),
                                  "application/json; charset=utf-8")
            self._send(200, b'{"ok":true}', "application/json")

        def log_message(self, *a):                  # тише в консоли
            pass

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"\nОткрой: http://127.0.0.1:{port}")
    print(f"Вердикты дописываются в {LABELS_CSV} сразу при нажатии.")
    print("Остановить: Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    finally:
        srv.server_close()


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
