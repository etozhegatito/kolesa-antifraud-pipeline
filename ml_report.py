# -*- coding: utf-8 -*-
"""
ml_report.py — красивый HTML-отчёт по модели цены (авто-тема).

Формирует самодостаточный data/eda/ml_report.html (открой в браузере):
  • «прожектор» на СЛУЧАЙНУЮ машину каждый запуск: оценка модели vs факт;
  • крупные карточки метрик (R², MAPE, MAE);
  • что двигает цену (важность признаков);
  • свежая выборка предсказаний.
Всё inline (CSS+SVG) — файл открывается без интернета.

Запуск: python ml_report.py   (офлайн, только Postgres)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from data_quality import scrub_junk_mileage
from train_price_model import CAT_FEATURES, FEATURES, cross_validate, load

OUT = "data/eda/ml_report.html"

CAR_SVG = """
<svg viewBox="0 0 680 250" class="car" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#5fb0ff"/><stop offset="1" stop-color="#1b2740"/>
    </linearGradient>
  </defs>
  <path fill="url(#body)" d="M40,178 L104,176 C126,150 150,126 196,110
    C244,93 296,88 348,88 L444,90 C500,94 540,120 576,150 L636,164
    C658,169 658,182 636,186 L120,190 C70,190 44,188 40,178 Z"/>
  <path fill="#0b0e14" opacity=".75" d="M214,112 C252,96 300,92 346,92
    L410,94 L432,140 L200,140 C202,128 206,120 214,112 Z"/>
  <line x1="330" y1="93" x2="330" y2="140" stroke="#0b0e14" stroke-width="6"/>
  <circle cx="200" cy="188" r="40" fill="#0b0e14" stroke="#ffd166" stroke-width="5"/>
  <circle cx="200" cy="188" r="15" fill="#2a2e3a"/>
  <circle cx="516" cy="188" r="40" fill="#0b0e14" stroke="#ffd166" stroke-width="5"/>
  <circle cx="516" cy="188" r="15" fill="#2a2e3a"/>
</svg>
"""

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Оценка цены авто · kolesa.kz</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #e8ecf3;
    background: radial-gradient(1200px 600px at 80% -10%, #1d3355 0%, transparent 60%),
                radial-gradient(900px 500px at 0% 10%, #2a1d3a 0%, transparent 55%),
                #0b0e14; min-height: 100vh; padding: 32px 20px; }
  .wrap { max-width: 1060px; margin: 0 auto; }
  header { display: flex; align-items: center; gap: 22px; margin-bottom: 26px; }
  .car { width: 260px; flex: 0 0 260px; filter: drop-shadow(0 10px 30px rgba(95,176,255,.25)); }
  h1 { font-size: 30px; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(90deg, #ffd166, #5fb0ff); -webkit-background-clip: text;
    background-clip: text; color: transparent; }
  header p { color: #9aa7bd; margin-top: 6px; font-size: 14px; }
  .card { background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px; padding: 22px 24px; backdrop-filter: blur(6px);
    box-shadow: 0 18px 40px rgba(0,0,0,.35); }
  .spot { display: flex; align-items: center; justify-content: space-between;
    gap: 24px; margin-bottom: 22px; flex-wrap: wrap; }
  .spot .who { color: #9aa7bd; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
  .spot .name { font-size: 26px; font-weight: 700; margin: 4px 0 2px; }
  .spot .spec { color: #aab6c9; font-size: 14px; }
  .price { text-align: right; }
  .price .big { font-size: 46px; font-weight: 800; line-height: 1;
    background: linear-gradient(90deg, #5fb0ff, #7c9cff); -webkit-background-clip: text;
    background-clip: text; color: transparent; }
  .price .lbl { color: #9aa7bd; font-size: 13px; }
  .badge { display: inline-block; margin-top: 10px; padding: 5px 12px; border-radius: 999px;
    font-size: 13px; font-weight: 600; }
  .b-low { background: rgba(79,163,255,.16); color: #7cc0ff; }
  .b-high { background: rgba(255,209,102,.16); color: #ffd166; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 22px; }
  .stat .v { font-size: 30px; font-weight: 800;
    background: linear-gradient(90deg, #ffd166, #ff9d6b); -webkit-background-clip: text;
    background-clip: text; color: transparent; }
  .stat .k { color: #9aa7bd; font-size: 13px; margin-top: 4px; }
  h2 { font-size: 16px; font-weight: 700; margin-bottom: 14px; color: #cdd6e6; }
  .bar { display: grid; grid-template-columns: 150px 1fr; align-items: center;
    gap: 12px; margin-bottom: 9px; font-size: 13px; }
  .track { height: 12px; background: rgba(255,255,255,.06); border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #5fb0ff, #ffd166); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.06); }
  th { color: #9aa7bd; font-weight: 600; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  footer { color: #6b7688; font-size: 12px; margin-top: 24px; text-align: center; }
</style></head><body><div class="wrap">
  <header>CAR_SVG<div>
    <h1>Оценка цены авто</h1>
    <p>kolesa.kz · Алматы · модель CatBoost на __N__ объявлениях (чистые)</p>
  </div></header>

  <div class="card spot">
    <div><div class="who">🎯 Прожектор · случайная машина</div>
      <div class="name">__CAR__</div><div class="spec">__SPEC__</div>
      <div class="badge __BCLS__">__VERDICT__</div></div>
    <div class="price"><div class="lbl">оценка модели</div>
      <div class="big">__EST__ млн ₸</div>
      <div class="lbl" style="margin-top:8px">в объявлении: __ACT__ млн ₸</div></div>
  </div>

  <div class="stats">__STATS__</div>

  <div class="card"><h2>Что двигает цену (важность признаков)</h2>__BARS__</div>

  <div class="card" style="margin-top:16px"><h2>Свежая выборка: оценка vs факт</h2>
    <table><tr><th>машина</th><th class="num">факт</th><th class="num">оценка</th>
    <th class="num">ошибка</th></tr>__ROWS__</table></div>

  <footer>Сгенерировано ml_report.py · каждый запуск — новая случайная машина ·
    метрики честные (out-of-fold 5-fold CV)</footer>
</div></body></html>"""


def _fit(X, y):
    m = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=8,
                          loss_function="RMSE", random_seed=42, verbose=False)
    m.fit(Pool(X, y, cat_features=CAT_FEATURES))
    return m


def _stat(v, k):
    return f'<div class="card stat"><div class="v">{v}</div><div class="k">{k}</div></div>'


def main():
    df = load()
    df = df[df["price_tenge"] > 0].copy()
    df["log_price"] = np.log(df["price_tenge"])
    df, _ = scrub_junk_mileage(df)
    clean = df[df["is_suspicious"] == 0]
    X, y = clean[FEATURES], clean["log_price"]

    r2, mae, mape = cross_validate(X, y).mean(axis=0)     # честные метрики
    model = _fit(X, y)

    # прожектор — случайная машина
    car = clean.sample(1).iloc[0]
    Xc = clean.loc[[car.name], FEATURES].copy()
    for c in CAT_FEATURES:
        Xc[c] = Xc[c].astype(str)
    est = float(np.exp(model.predict(Xc)[0]))
    act = float(car["price_tenge"])
    diff = (act - est) / est * 100
    verdict = (f"дешевле оценки на {abs(diff):.0f}%" if diff < 0
               else f"дороже оценки на {diff:.0f}%")
    bcls = "b-low" if diff < 0 else "b-high"
    mil = "—" if pd.isna(car["mileage_km"]) else f"{int(car['mileage_km']):,} км".replace(",", " ")
    ev = "" if pd.isna(car["engine_volume"]) else f'{car["engine_volume"]:.1f}л '
    spec = f'{int(car["year"])} · {car["engine_type"]} {ev}· {car["transmission"]} · {mil}'

    # важность
    imp = sorted(zip(FEATURES, model.get_feature_importance()), key=lambda t: -t[1])[:8]
    mx = imp[0][1]
    bars = "".join(f'<div class="bar"><span>{f}</span>'
                   f'<div class="track"><div class="fill" style="width:{v/mx*100:.0f}%"></div></div></div>'
                   for f, v in imp)

    # свежая выборка
    samp = clean.sample(min(6, len(clean)))
    Xs = samp[FEATURES].copy()
    for c in CAT_FEATURES:
        Xs[c] = Xs[c].astype(str)
    samp = samp.assign(pred=np.exp(model.predict(Xs)))
    rows = ""
    for _, r in samp.iterrows():
        err = abs(r["pred"] - r["price_tenge"]) / r["price_tenge"] * 100
        rows += (f'<tr><td>{r["brand"]} {r["model"]} {int(r["year"])}</td>'
                 f'<td class="num">{r["price_tenge"]/1e6:.1f}М</td>'
                 f'<td class="num">{r["pred"]/1e6:.1f}М</td>'
                 f'<td class="num">{err:.0f}%</td></tr>')

    stats = (_stat(f"{r2:.3f}", "R² (log) — точность")
             + _stat(f"{mape:.0f}%", "MAPE — средняя ошибка")
             + _stat(f"{mae/1e6:.1f}М", "MAE — ₸ мимо")
             + _stat(f"{len(X):,}".replace(",", " "), "машин в обучении"))

    html = (PAGE.replace("CAR_SVG", CAR_SVG)
            .replace("__N__", f"{len(X):,}".replace(",", " "))
            .replace("__CAR__", f'{car["brand"]} {car["model"]}')
            .replace("__SPEC__", spec).replace("__VERDICT__", verdict).replace("__BCLS__", bcls)
            .replace("__EST__", f"{est/1e6:.1f}").replace("__ACT__", f"{act/1e6:.1f}")
            .replace("__STATS__", stats).replace("__BARS__", bars).replace("__ROWS__", rows))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Отчёт → {OUT}   (прожектор: {car['brand']} {car['model']} "
          f"{int(car['year'])}, оценка {est/1e6:.1f}М vs факт {act/1e6:.1f}М)")


if __name__ == "__main__":
    main()
