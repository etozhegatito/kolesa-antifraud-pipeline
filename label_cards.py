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
    """Цена в читаемом виде; пусто — прочерк."""
    if pd.isna(v) or v is None:
        return "—"
    return f"{float(v)/1e6:.2f}М ₸" if float(v) >= 1e5 else f"{int(float(v)):,} ₸"


def fmt(v) -> str:
    """Значение для таблички: NaN/None/пустое → прочерк, иначе экранируем."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    if isinstance(v, float) and float(v).is_integer():
        v = int(v)
    return html.escape(str(v))


def price_verdict_hint(row) -> str:
    """Кросс-чек с avgPrice самого kolesa — валидатор, НЕ признак модели."""
    avg = row.get("kolesa_avg_price")
    price = row.get("price_tenge")
    if pd.isna(avg) or avg is None or float(avg) <= 0 or pd.isna(price):
        return ""
    ratio = float(price) / float(avg)
    if ratio < 0.6:
        return (f"kolesa считает средней {money(avg)} — это <b>{ratio*100:.0f}%</b> "
                "от среднего, сильно дешевле рынка")
    if ratio > 1.4:
        return (f"kolesa считает средней {money(avg)} — цена ВЫШЕ рынка "
                f"({ratio*100:.0f}%), приманкой быть не может")
    return (f"kolesa считает средней {money(avg)} — цена в норме "
            f"({ratio*100:.0f}% от среднего)")


def card_html(row) -> str:
    """Одна карточка объявления."""
    aid = html.escape(str(row["ad_id"]))
    reasons = [r for r in str(row.get("suspicion_reasons") or "").split(";") if r]
    reasons = [p for r in reasons for p in r.split("|")]
    dead = row.get("status") in ("archived", "deleted")
    badge = row.get("page_status_badge")
    has_badge = badge not in (None, "-", "") and not pd.isna(badge)

    photos = row["photos"]
    imgs = "".join(
        f'<a href="{html.escape(u)}" target="_blank">'
        f'<img loading="lazy" src="{html.escape(u)}" alt="фото {i+1}"></a>'
        for i, u in enumerate(photos)) or \
        '<div class="nophoto">фото-URL не сохранены</div>'

    help_blocks = ""
    for r in reasons:
        if r in FLAG_HELP:
            what, fr, lg = FLAG_HELP[r]
            help_blocks += (
                f'<div class="help"><div class="hflag">{html.escape(r)}</div>'
                f'<div class="hwhat">{html.escape(what)}</div>'
                f'<div class="hfraud">{fr}</div>'
                f'<div class="hlegit">{lg}</div></div>')

    facts = [
        ("Цена", money(row.get("price_tenge"))),
        ("Год", fmt(row.get("year"))),
        ("Пробег (листинг)", fmt(row.get("mileage_km"))),
        ("Пробег (страница)", fmt(row.get("page_mileage_km"))),
        ("Двигатель", f'{fmt(row.get("engine_volume"))} / {fmt(row.get("engine_type"))}'),
        ("Коробка", fmt(row.get("transmission"))),
        ("Кузов", fmt(row.get("body_type"))),
        ("Цвет", fmt(row.get("color"))),
        ("Привод / руль", f'{fmt(row.get("drive"))} / {fmt(row.get("steering"))}'),
        ("Растаможен", fmt(row.get("customs_cleared"))),
        ("Состояние (страница)", fmt(row.get("page_condition"))),
        ("VIN указан", fmt(row.get("has_vin"))),
        ("price_z", fmt(round(float(row["price_z"]), 2)
                        if pd.notna(row.get("price_z")) else None)),
        ("Просмотров", fmt(row.get("views_count"))),
        ("Размещено", fmt(row.get("posted_date"))),
    ]
    facts_html = "".join(f'<div class="f"><span>{k}</span><b>{v}</b></div>'
                         for k, v in facts)

    texts = ""
    for label, key in [("Описание (листинг)", "description"),
                       ("Комментарий продавца", "seller_comment"),
                       ("Опции", "options_text")]:
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
            texts += (f'<div class="txt"><div class="tl">{label}</div>'
                      f'<div class="tb">{html.escape(str(v))}</div></div>')
    if not texts:
        texts = '<div class="nophoto">текста нет вообще — решай по фото и цене</div>'

    dmg = row.get("damage_keywords")
    dmg_html = ""
    if dmg and not (isinstance(dmg, float) and pd.isna(dmg)) and str(dmg).strip():
        dmg_html = (f'<div class="dmg">⚠ damage-слова в тексте: '
                    f'<b>{html.escape(str(dmg))}</b> — это в пользу legit '
                    f'(продавец САМ раскрыл проблему)</div>')
    if has_badge:
        dmg_html += (f'<div class="dmg">⚠ бейдж сайта: '
                     f'<b>{html.escape(str(badge))}</b> — сайт сам помечает '
                     f'машину как проблемную, дешевизна объяснена → legit</div>')

    hint = price_verdict_hint(row)
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""

    ev = row.get("existing_verdict")
    ev_html = (f'<div class="done">уже размечено: <b>{html.escape(str(ev))}</b> '
               f'(можно перепроверить)</div>'
               if ev and not pd.isna(ev) else "")

    status_html = (f'<span class="dead">{html.escape(str(row.get("status")))} — '
                   f'страницы больше нет, но фото ниже живы</span>' if dead else
                   f'<a class="live" href="https://kolesa.kz/a/show/{aid}" '
                   f'target="_blank">открыть на kolesa ↗ (тратит лимит IP!)</a>')

    return f"""
<section class="card" id="ad{aid}">
  <header>
    <div>
      <h2>{html.escape(str(row.get("brand") or ""))} {html.escape(str(row.get("model") or ""))}
          <span class="yr">{fmt(row.get("year"))}</span></h2>
      <div class="meta">id {aid} · {status_html}</div>
    </div>
    <div class="flags">{"".join(f'<span class="flag">{html.escape(r)}</span>' for r in reasons)}</div>
  </header>
  {ev_html}
  <div class="gal">{imgs}</div>
  {hint_html}
  {dmg_html}
  <div class="facts">{facts_html}</div>
  {texts}
  <div class="helps">{help_blocks}</div>
  <div class="actions">
    <button class="bfraud" onclick="pick('{aid}','fraud')">fraud (обман)</button>
    <button class="blegit" onclick="pick('{aid}','legit')">legit (честно)</button>
    <button class="bunk" onclick="pick('{aid}','unknown')">unknown (не понять)</button>
    <input class="cmt" id="c{aid}" placeholder="почему (попадёт в comment)">
    <span class="picked" id="p{aid}"></span>
  </div>
</section>"""


def build(rows: pd.DataFrame) -> str:
    cards = "".join(card_html(r) for _, r in rows.iterrows())
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
TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Карточки для разметки</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0f1115;color:#e6e8ee;
     font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 6px}
.sub{color:#9aa3b2;margin-bottom:18px}
.safe{background:#12281a;border:1px solid #2c6b3f;border-radius:10px;
      padding:12px 14px;margin-bottom:18px;color:#b9e7c6}
.basket{position:sticky;top:0;z-index:9;background:#161a22;
        border:1px solid #2a3140;border-radius:12px;padding:14px;margin-bottom:22px}
.basket textarea{width:100%;height:120px;background:#0f1115;color:#e6e8ee;
        border:1px solid #2a3140;border-radius:8px;padding:10px;
        font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.basket .row{display:flex;gap:10px;align-items:center;margin-bottom:8px}
.basket b{font-size:15px}
button{background:#222836;color:#e6e8ee;border:1px solid #38415a;
       border-radius:8px;padding:8px 14px;cursor:pointer;font-size:14px}
button:hover{background:#2b3346}
.bfraud{border-color:#7a2b2b;background:#2a1618}
.blegit{border-color:#2c6b3f;background:#12281a}
.card{background:#151922;border:1px solid #242b39;border-radius:14px;
      padding:18px;margin-bottom:26px}
.card header{display:flex;justify-content:space-between;gap:16px;
             align-items:flex-start;margin-bottom:12px}
h2{font-size:19px;margin:0}
.yr{color:#9aa3b2;font-weight:400}
.meta{color:#9aa3b2;font-size:13px;margin-top:4px}
.dead{color:#ffb454}
.live{color:#6fa8ff}
.flags{display:flex;flex-wrap:wrap;gap:6px;justify-content:flex-end}
.flag{background:#2a1618;border:1px solid #7a2b2b;color:#ffb3b3;
      border-radius:999px;padding:3px 10px;font-size:12px}
.gal{display:flex;gap:8px;overflow-x:auto;padding-bottom:8px;margin-bottom:12px}
.gal img{height:190px;border-radius:10px;border:1px solid #242b39;display:block}
.nophoto{color:#9aa3b2;font-style:italic;padding:8px 0}
.hint{background:#12203a;border:1px solid #2b4a7a;border-radius:8px;
      padding:10px 12px;margin-bottom:10px;font-size:14px;color:#cfe0ff}
.dmg{background:#2b2410;border:1px solid #6b5a1f;border-radius:8px;
     padding:10px 12px;margin-bottom:10px;font-size:14px;color:#f2e2b0}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
       gap:6px 16px;margin:12px 0}
.f{display:flex;justify-content:space-between;border-bottom:1px solid #1e2431;
   padding:4px 0;font-size:13px}
.f span{color:#9aa3b2}
.txt{margin:10px 0}
.tl{color:#9aa3b2;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tb{background:#0f1115;border:1px solid #242b39;border-radius:8px;
    padding:10px 12px;white-space:pre-wrap;margin-top:4px}
.helps{margin-top:14px}
.help{border-left:3px solid #38415a;padding:8px 0 8px 12px;margin:10px 0}
.hflag{font:13px ui-monospace,Menlo,monospace;color:#9aa3b2}
.hwhat{margin:2px 0 6px}
.hfraud{color:#ffb3b3;font-size:14px}
.hlegit{color:#b9e7c6;font-size:14px;margin-top:3px}
.actions{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:14px;
         border-top:1px solid #242b39;padding-top:14px}
.cmt{flex:1;min-width:220px;background:#0f1115;color:#e6e8ee;
     border:1px solid #2a3140;border-radius:8px;padding:8px 10px}
.picked{color:#b9e7c6;font-size:14px}
.done{background:#12281a;border:1px solid #2c6b3f;border-radius:8px;
      padding:8px 12px;margin-bottom:10px;color:#b9e7c6;font-size:14px}
a{color:#6fa8ff}
</style>
<div class="wrap">
<h1>Карточки для разметки — __N__ объявлений</h1>
<div class="sub">из них __NDEAD__ с мёртвой страницей (фото всё равно видно) ·
  __NDONE__ уже размечено</div>

<div class="safe">
  <b>Эта страница не ходит на kolesa.kz.</b> Фото грузятся с CDN
  (kcdn.kz — другой хост), весь текст и цифры взяты из локальной базы.
  Разметка здесь <b>не тратит</b> суточный лимит запросов к kolesa.
  Ссылка «открыть на kolesa» — единственное исключение, она тратит.
</div>

<div class="basket">
  <div class="row"><b>Собранные вердикты:</b>
    <span id="cnt">0</span>
    <button onclick="copyAll()">копировать всё</button>
    <button onclick="clearAll()">очистить</button>
  </div>
  <textarea id="out" readonly placeholder="Нажимай fraud/legit/unknown в карточках — строки соберутся здесь. Потом «копировать всё» и допиши в конец __LABELS__ (файл только ДОПИСЫВАЕТСЯ, не перезаписывается)."></textarea>
</div>

__CARDS__
</div>
<script>
// Формат строки — как в manual_labels.csv; clean.py читает только ad_id и
// verdict, остальные колонки можно оставить пустыми.
const picks = new Map();
function esc(s){ return /[",\\n]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s; }
function pick(id, verdict){
  const c = document.getElementById('c'+id).value.trim();
  picks.set(id, verdict + ',' + esc(c));
  document.getElementById('p'+id).textContent = '→ ' + verdict;
  render();
}
function render(){
  const lines = [];
  for (const [id, v] of picks) lines.push(id + ',,,,,,,,' + v);
  document.getElementById('out').value = lines.join('\\n');
  document.getElementById('cnt').textContent = picks.size;
}
function copyAll(){
  const t = document.getElementById('out');
  if (!t.value) return;
  navigator.clipboard.writeText(t.value + '\\n');
}
function clearAll(){ picks.clear(); render(); }
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
