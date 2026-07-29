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

Запуск:  python label_cards.py            → data/eda/label_cards.html
         python label_cards.py --all      → включить и residual-кандидатов
                                            из labeling_queue.csv, не только
                                            правиловых подозрительных

Вердикты НЕ пишутся отсюда автоматически: страница только собирает
строки, которые ты сам вставишь в data/manual_labels.csv (журнал
append-only — правило проекта №1, дописывается только руками).
"""

import pathlib as _p
_expected = "label_cards.py"
if _p.Path(__file__).name != _expected:
    raise SystemExit(f"ОШИБКА: этот код — {_expected}, а файл называется "
                     f"{_p.Path(__file__).name}.")

import html
import json
import sys
from pathlib import Path

import pandas as pd

from db import get_engine

OUT_HTML   = "data/eda/label_cards.html"
QUEUE_CSV  = "data/eda/labeling_queue.csv"
LABELS_CSV = "data/manual_labels.csv"

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
    if include_queue and Path(QUEUE_CSV).exists():
        q = pd.read_csv(QUEUE_CSV, dtype={"ad_id": str})
        ids |= set(q["ad_id"])
    rows = cd[cd["ad_id"].isin(ids)].copy()

    # Доп. поля со страницы, которых нет в clean_data.
    enr = pd.read_sql("SELECT ad_id, options_text, page_condition, has_vin, "
                      "fetched_at FROM enriched", eng, dtype={"ad_id": str})
    rows = rows.merge(enr, on="ad_id", how="left")

    photos = pd.read_sql("SELECT ad_id, position, url FROM photos", eng,
                         dtype={"ad_id": str})
    photos = photos[photos["url"].fillna("").str.startswith("http")]
    gal = (photos.sort_values(["ad_id", "position"])
           .groupby("ad_id")["url"].apply(list))
    rows["photos"] = rows["ad_id"].map(gal)
    rows["photos"] = rows["photos"].apply(lambda v: v if isinstance(v, list) else [])

    # Уже размеченные помечаем, но НЕ выкидываем: удобно перепроверить.
    if Path(LABELS_CSV).exists():
        lab = pd.read_csv(LABELS_CSV, dtype={"ad_id": str})
        done = lab[lab["verdict"].isin(["fraud", "legit"])]
        rows["existing_verdict"] = rows["ad_id"].map(
            dict(zip(done["ad_id"], done["verdict"])))
    else:
        rows["existing_verdict"] = None
    return rows.sort_values(["existing_verdict", "price_z"], na_position="first")


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


def card_html(row, idx: int) -> str:
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

    photos = row["photos"]
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
<article class="card" id="ad{aid}" data-id="{aid}" data-idx="{idx}">
  <header>
    <div class="ttl">
      <h2>{title} <span class="yr">{fmt(row.get("year"))}</span></h2>
      <div class="meta"><span class="num">#{idx + 1}</span> id {aid} · {status_html}</div>
    </div>
    <div class="flags">
      {"".join(f'<span class="flag">{html.escape(r)}</span>' for r in reasons)}
    </div>
  </header>
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


def build(rows: pd.DataFrame) -> str:
    cards = "".join(card_html(r, i)
                    for i, (_, r) in enumerate(rows.iterrows()))
    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_done = int(rows["existing_verdict"].notna().sum())
    return (TEMPLATE
            .replace("__CARDS__", cards)
            .replace("__N__", str(len(rows)))
            .replace("__NDEAD__", str(n_dead))
            .replace("__NDONE__", str(n_done))
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
.picked{color:var(--legit); font-size:.875rem; font-weight:500}

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
    <button class="tbtn" id="filter">скрыть размеченные</button>
    <button class="tbtn" id="theme">тема</button>
  </div>
</div>

<p class="lede">__N__ объявлений, из них __NDEAD__ с мёртвой страницей — фото у них
всё равно видно. Уже размечено ранее: __NDONE__.</p>

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
  <summary>Собранные вердикты — <span id="cnt2">0</span></summary>
  <div class="in">
    <textarea id="out" readonly placeholder="Нажимай fraud / legit / unknown в карточках — строки соберутся здесь."></textarea>
    <div class="row">
      <button class="tbtn" id="copy">копировать всё</button>
      <button class="tbtn" id="clear">очистить</button>
    </div>
    <p class="hintline">Скопированные строки нужно <b>дописать в конец</b>
      __LABELS__. Журнал только дополняется и никогда не перезаписывается:
      прошлые вердикты остаются валидными, даже если объявление ушло из очереди.
      После этого — <span class="mono">python clean.py</span> и
      <span class="mono">python evaluate_detector.py</span>.</p>
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
const cards = Array.from(document.querySelectorAll('.card'));
const picks = new Map();
let cur = 0;

/* Кавычки и переводы строк экранируются по правилам CSV, иначе комментарий
   с запятой сдвинул бы колонки журнала. */
function esc(s){ return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }

function render(){
  const lines = [];
  for (const [id, v] of picks) lines.push(id + ',,,,,,,,' + v);
  document.getElementById('out').value = lines.join('\n');
  document.getElementById('cnt').textContent = picks.size;
  document.getElementById('cnt2').textContent = picks.size;
  document.getElementById('bar').style.width =
    (cards.length ? picks.size / cards.length * 100 : 0) + '%';
}

function setVerdict(card, v){
  if (!card) return;
  const cmt = card.querySelector('.cmt').value.trim();
  picks.set(card.dataset.id, v + ',' + esc(cmt));
  card.dataset.verdict = v;
  card.querySelector('.picked').textContent = '→ ' + v;
  render();
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
  picks.clear();
  cards.forEach(c => { delete c.dataset.verdict; c.querySelector('.picked').textContent = ''; });
  render();
};
document.getElementById('filter').onclick = e => {
  document.body.classList.toggle('hide-done');
  const on = document.body.classList.contains('hide-done');
  e.target.classList.toggle('on', on);
  e.target.textContent = on ? 'показать все' : 'скрыть размеченные';
};
document.getElementById('theme').onclick = () => {
  const root = document.documentElement;
  const now = root.dataset.theme ||
    (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  root.dataset.theme = now === 'dark' ? 'light' : 'dark';
};

focusCard(0);
render();
</script>
"""


def main():
    include_queue = "--all" in sys.argv
    rows = load_rows(include_queue)
    if rows.empty:
        print("Нечего размечать: подозрительных нет.")
        return
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(build(rows), encoding="utf-8")

    n_dead = int(rows["status"].isin(["archived", "deleted"]).sum())
    n_photo = int(rows["photos"].apply(bool).sum())
    print(f"Карточек: {len(rows)} (мёртвых страниц: {n_dead}, "
          f"с фото: {n_photo})")
    print(f"→ {OUT_HTML}")
    print("Открой в браузере, размечай, потом «копировать всё» и допиши "
          f"строки в конец {LABELS_CSV} (ТОЛЬКО дописывать!).")
    print("kolesa.kz при этом не запрашивается — лимит не тратится.")


if __name__ == "__main__":
    main()
