# -*- coding: utf-8 -*-
"""HTML-страницы веб-интерфейса.

Страницы собираются в Python и отдаются целиком, без шаблонизатора и без
внешних CDN: приложение локальное, а лишняя зависимость ради двух страниц
не окупается. Оформление — то же, что в карточках разметки: системные
шрифты, светлая и тёмная тема по настройке системы.
"""

CSS = """
:root{
  --bg:#0c0f16; --surface:#131824; --surface2:#1a2030; --line:#232b3d;
  --text:#e7eaf2; --muted:#8f98ab; --accent:#7aa7ff; --accent-bg:#14203a;
  --ok:#7fe0a5; --ok-bg:#102319; --warn:#ffc470; --warn-bg:#241c0e;
  --bad:#ff8b8b; --bad-bg:#2a1518;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f6f7f9; --surface:#fff; --surface2:#f2f4f7; --line:#e2e6ed;
    --text:#161a22; --muted:#5f6773; --accent:#2563c9; --accent-bg:#eaf1ff;
    --ok:#1a7a48; --ok-bg:#eaf7ef; --warn:#8a5a00; --warn-bg:#fdf3e0;
    --bad:#b4232c; --bad-bg:#fdeeef;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;
  font-variant-numeric:tabular-nums}
.wrap{max-width:900px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:1.6rem;font-weight:600;letter-spacing:-.02em;margin:0 0 6px}
h2{font-size:1.1rem;font-weight:600;margin:26px 0 10px}
.sub{color:var(--muted);margin:0 0 24px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:20px;margin-bottom:18px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
label{display:block;font-size:.8125rem;color:var(--muted);margin-bottom:4px}
input,select,textarea{width:100%;background:var(--bg);color:var(--text);
  border:1px solid var(--line);border-radius:9px;padding:9px 11px;font:inherit}
textarea{min-height:80px;resize:vertical}
button{background:var(--accent);color:#fff;border:none;border-radius:10px;
  padding:11px 22px;font:inherit;font-weight:500;cursor:pointer;margin-top:16px}
button:hover{filter:brightness(1.08)}
.big{font-size:2.1rem;font-weight:600;letter-spacing:-.02em}
.range{color:var(--muted);font-size:.9375rem}
.row{display:flex;justify-content:space-between;gap:12px;padding:8px 0;
  border-bottom:1px solid var(--line);font-size:.9rem}
.row:last-child{border-bottom:none}
.bar{height:6px;border-radius:3px;background:var(--surface2);overflow:hidden;
  margin-top:4px}
.bar i{display:block;height:6px}
.up{background:var(--ok)} .down{background:var(--bad)}
.note{border-radius:9px;padding:11px 13px;margin-top:9px;font-size:.9rem}
.note.warn{background:var(--warn-bg);border:1px solid var(--line);color:var(--warn)}
.note.info{background:var(--accent-bg);border:1px solid var(--line)}
.muted{color:var(--muted);font-size:.8125rem}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th{text-align:left;color:var(--muted);font-weight:500;font-size:.8125rem;
  padding-bottom:6px}
td{padding:6px 0;border-top:1px solid var(--line)}
.hide{display:none}
"""

_NAV = """<div class="sub"><a href="/">← главная</a></div>"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style>
<div class="wrap">{body}</div>"""


def index_page() -> str:
    return _page("KZ Car Market", """
<h1>KZ Car Market</h1>
<p class="sub">Оценка автомобиля и разметка антифрода.</p>
<div class="card">
  <h2 style="margin-top:0"><a href="/estimate">Оценить машину →</a></h2>
  <p class="muted">Характеристики и описание — на выходе справедливая цена,
  диапазон, разбор «почему столько», позиция среди похожих объявлений и
  замечания к объявлению.</p>
</div>
<div class="card">
  <h2 style="margin-top:0"><a href="/label">Разметка вердиктов →</a></h2>
  <p class="muted">Решение по объявлению целиком: fraud, legit или unknown.
  Здесь вместе находятся подозрительные, кандидаты второй модели и случайный
  контроль — без него нельзя измерить пропуски детектора. Один объект здесь —
  одно объявление.</p>
</div>
<div class="card">
  <h2 style="margin-top:0"><a href="/damage">Разметка повреждений →</a></h2>
  <p class="muted">Решение по отдельной фотографии: обвести удар рамкой или
  отметить кадр целым, аварийным, разобранным либо неясным. Поэтому счётчик
  здесь другой: один автомобиль может дать несколько кадров.</p>
</div>
<p class="muted">Состояние модели: <a href="/api/health">/api/health</a> ·
Документация API: <a href="/api/docs">/api/docs</a></p>
""")


def estimate_page() -> str:
    body = _NAV + """
<h1>Оценка автомобиля</h1>
<p class="sub">Обязательны марка, модель и год — остальное уточняет оценку.</p>

<div class="card">
  <div class="grid">
    <div><label>Марка</label><input id="brand" value="Toyota"></div>
    <div><label>Модель</label><input id="model" value="Camry"></div>
    <div><label>Год выпуска</label><input id="year" type="number" value="2019"></div>
    <div><label>Пробег, км</label><input id="mileage_km" type="number" value="95000"></div>
    <div><label>Объём двигателя, л</label><input id="engine_volume" type="number" step="0.1" value="2.5"></div>
    <div><label>Топливо</label><select id="engine_type">
      <option>бензин</option><option>дизель</option><option>газ-бензин</option>
      <option>газ</option><option>гибрид</option><option>электро</option>
      </select></div>
    <div><label>Коробка</label><select id="transmission">
      <option>автомат</option><option>механика</option><option>вариатор</option>
      <option>робот</option><option>типтроник</option></select></div>
    <div><label>Кузов</label><select id="body_type">
      <option>седан</option><option>кроссовер</option><option>внедорожник</option>
      <option>минивэн</option><option>хэтчбек</option><option>универсал</option>
      <option>фургон</option><option>пикап</option><option>лифтбек</option>
      <option>купе</option><option>микроавтобус</option><option>кабриолет</option>
      <option>родстер</option></select></div>
    <div><label>Состояние</label><select id="condition">
      <option>б/у</option><option>новый</option></select></div>
    <div><label>Сколько фотографий</label><input id="photos_count" type="number" value="8"></div>
    <div><label>Ваша цена, ₸ (необязательно)</label><input id="asking_price" type="number" placeholder="напр. 11000000"></div>
  </div>
  <div style="margin-top:14px">
    <label>Описание для покупателя (необязательно)</label>
    <textarea id="text" placeholder="Один хозяин, обслужена у дилера…"></textarea>
  </div>
  <button onclick="run()">Оценить</button>
</div>

<div id="out" class="hide"></div>

<script>
function money(v){ return (v/1e6).toFixed(2).replace('.', ',') + ' млн ₸'; }
function esc(v){
  return String(v ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function run(){
  const ids = ['brand','model','year','mileage_km','engine_volume','engine_type',
               'transmission','body_type','condition','photos_count',
               'asking_price','text'];
  const car = {};
  ids.forEach(k => { const v = document.getElementById(k).value; if (v !== '') car[k] = v; });
  const out = document.getElementById('out');
  out.className = ''; out.innerHTML = '<div class="card">Считаю…</div>';
  const r = await fetch('/api/estimate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(car)});
  const d = await r.json();
  if (d.error){ out.innerHTML = '<div class="card note warn">Ошибка: '+esc(d.error)+'</div>'; return; }

  let h = '<div class="card"><div class="muted">Справедливая цена</div>'
        + '<div class="big">' + money(d.fair_price) + '</div>'
        + '<div class="range">вероятный диапазон ' + money(d.range_low)
        + ' — ' + money(d.range_high) + '</div>'
        + '<div class="muted" style="margin-top:10px">Модель обучена на '
        + d.trained_rows + ' машинах, средняя ошибка около '
        + d.model_mape_pct.toFixed(0) + '%. Это оценка цены ПУБЛИКАЦИИ, '
        + 'а не гарантия суммы сделки.</div></div>';

  if (d.position){
    const p = d.position;
    h += '<div class="card"><h2 style="margin-top:0">Ваша цена среди похожих</h2>'
      + '<div>' + esc(p.label) + ' — дешевле ' + p.percentile.toFixed(0)
      + '% из ' + p.n_similar + ' машин</div>'
      + '<div class="muted" style="margin-top:6px">Половина похожих стоит от '
      + money(p.p25) + ' до ' + money(p.p75) + '.</div>'
      + '<div class="note info" style="margin-top:10px">Это позиция среди '
      + 'выставленных цен, а не прогноз срока продажи: истории наблюдений '
      + 'пока мало, чтобы обещать сроки.</div></div>';
  }

  h += '<div class="card"><h2 style="margin-top:0">Почему столько</h2>';
  d.drivers.forEach(x => {
    const up = x.effect_pct >= 0;
    const w = Math.min(100, Math.abs(x.effect_pct));
    h += '<div class="row"><span>' + esc(x.feature) + ' <span class="muted">'
       + esc(x.value) + '</span></span><b>' + (up?'+':'') + x.effect_pct.toFixed(0)
       + '%</b></div><div class="bar"><i class="' + (up?'up':'down')
       + '" style="width:' + w + '%"></i></div>';
  });
  h += '<div class="muted" style="margin-top:10px">Вклад каждой характеристики '
     + 'в итоговую цену именно этой машины.</div></div>';

  if (d.warnings.length){
    h += '<div class="card"><h2 style="margin-top:0">Что улучшить в объявлении</h2>';
    d.warnings.forEach(w => h += '<div class="note warn">' + esc(w) + '</div>');
    h += '</div>';
  }

  if (d.similar.length){
    h += '<div class="card"><h2 style="margin-top:0">Похожие объявления</h2>'
       + '<table><tr><th>Машина</th><th>Год</th><th>Пробег</th><th>Цена</th></tr>';
    d.similar.forEach(s => {
      h += '<tr><td>' + esc(s.brand) + ' ' + esc(s.model) + '</td><td>' + esc(s.year)
         + '</td><td>' + (s.mileage_km ? Math.round(s.mileage_km).toLocaleString('ru') : '—')
         + '</td><td>' + money(s.price_tenge) + '</td></tr>';
    });
    h += '</table></div>';
  }
  out.innerHTML = h;
}
</script>"""
    return _page("Оценка автомобиля", body)
