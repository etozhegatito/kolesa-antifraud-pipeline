# -*- coding: utf-8 -*-
"""HTML pages for the web interface.

The pages are assembled in Python and returned without a template engine or
external CDN. This keeps the local application self-contained. The visual
language matches the labelling tools and follows the system light/dark theme.
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

_NAV = """<div class="sub"><a href="/">← Home</a></div>"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style>
<div class="wrap">{body}</div>"""


def index_page() -> str:
    return _page(
        "KZ Auto Market Intelligence",
        """
<h1>KZ Auto Market Intelligence</h1>
<p class="sub">Vehicle valuation, market anomaly review, and visual condition analysis.</p>
<div class="card">
  <h2 style="margin-top:0"><a href="/estimate">Estimate a vehicle →</a></h2>
  <p class="muted">Enter the vehicle specifications and description to get a
  fair listing-price estimate, an uncertainty range, an explanation, market
  position, comparable listings, and listing-quality checks.</p>
</div>
<div class="card">
  <h2 style="margin-top:0"><a href="/label">Review market anomalies →</a></h2>
  <p class="muted">Assign a manual fraud, legitimate, or unknown verdict.
  The queue combines rule-based alerts, residual-model candidates, and a
  random control sample. A candidate is not an accusation. One item is one listing.</p>
</div>
<div class="card">
  <h2 style="margin-top:0"><a href="/price-review">Review below-5M condition →</a></h2>
  <p class="muted">Inspect a fixed 50-listing pilot with locally stored photos.
  Label vehicle state, price meaning, and whether the evidence came from text,
  photos, both, or neither. Predictions stay hidden to avoid anchoring.</p>
</div>
<div class="card">
  <h2 style="margin-top:0"><a href="/damage">Label vehicle damage →</a></h2>
  <p class="muted">Review individual photos: draw boxes around local impact
  damage or mark a frame as intact, wrecked, dismantled, or unclear. Counts
  differ from anomaly review because one listing can contain several frames.</p>
</div>
<p class="muted">Model status: <a href="/api/health">/api/health</a> ·
API documentation: <a href="/api/docs">/api/docs</a></p>
""",
    )


def estimate_page() -> str:
    body = (
        _NAV
        + """
<h1>Vehicle price estimate</h1>
<p class="sub">Brand, model, and year are required. Other fields improve the estimate.</p>

<div class="card">
  <div class="grid">
    <div><label>Brand</label><input id="brand" value="Toyota"></div>
    <div><label>Model</label><input id="model" value="Camry"></div>
    <div><label>Model year</label><input id="year" type="number" value="2019"></div>
    <div><label>Mileage, km</label><input id="mileage_km" type="number" value="95000"></div>
    <div><label>Engine displacement, L</label><input id="engine_volume" type="number" step="0.1" value="2.5"></div>
    <div><label>Fuel</label><select id="engine_type">
      <option value="бензин">Petrol</option><option value="дизель">Diesel</option>
      <option value="газ-бензин">Petrol/LPG</option><option value="газ">LPG</option>
      <option value="гибрид">Hybrid</option><option value="электро">Electric</option>
      </select></div>
    <div><label>Transmission</label><select id="transmission">
      <option value="автомат">Automatic</option><option value="механика">Manual</option>
      <option value="вариатор">CVT</option><option value="робот">Automated manual</option>
      <option value="типтроник">Tiptronic</option></select></div>
    <div><label>Body style</label><select id="body_type">
      <option value="седан">Sedan</option><option value="кроссовер">Crossover</option>
      <option value="внедорожник">SUV</option><option value="минивэн">Minivan</option>
      <option value="хэтчбек">Hatchback</option><option value="универсал">Wagon</option>
      <option value="фургон">Van</option><option value="пикап">Pickup</option>
      <option value="лифтбек">Liftback</option><option value="купе">Coupe</option>
      <option value="микроавтобус">Minibus</option><option value="кабриолет">Convertible</option>
      <option value="родстер">Roadster</option></select></div>
    <div><label>Condition</label><select id="condition">
      <option value="б/у">Used</option><option value="новый">New</option></select></div>
    <div><label>Number of photos</label><input id="photos_count" type="number" value="8"></div>
    <div><label>Your asking price, ₸ (optional)</label><input id="asking_price" type="number" placeholder="e.g. 11000000"></div>
  </div>
  <div style="margin-top:14px">
    <label>Listing description (optional)</label>
    <textarea id="text" placeholder="One owner, dealer-maintained…"></textarea>
  </div>
  <button onclick="run()">Estimate price</button>
</div>

<div id="out" class="hide"></div>

<script>
function money(v){ return (v/1e6).toFixed(2) + 'M ₸'; }
function esc(v){
  return String(v ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
const FEATURE_LABELS = {
  brand:'Brand', model:'Model', age:'Age', mileage_km:'Mileage',
  engine_volume:'Engine displacement', engine_type:'Fuel',
  transmission:'Transmission', body_type:'Body style', condition:'Condition',
  photos_count:'Photo count'
};
const VALUE_LABELS = {
  'бензин':'Petrol', 'дизель':'Diesel', 'газ-бензин':'Petrol/LPG',
  'газ':'LPG', 'гибрид':'Hybrid', 'электро':'Electric',
  'автомат':'Automatic', 'механика':'Manual', 'вариатор':'CVT',
  'робот':'Automated manual', 'типтроник':'Tiptronic',
  'седан':'Sedan', 'кроссовер':'Crossover', 'внедорожник':'SUV',
  'минивэн':'Minivan', 'хэтчбек':'Hatchback', 'универсал':'Wagon',
  'фургон':'Van', 'пикап':'Pickup', 'лифтбек':'Liftback', 'купе':'Coupe',
  'микроавтобус':'Minibus', 'кабриолет':'Convertible', 'родстер':'Roadster',
  'б/у':'Used', 'новый':'New'
};
function featureLabel(v){ return FEATURE_LABELS[v] || v; }
function valueLabel(v){ return VALUE_LABELS[v] || v; }

async function run(){
  const ids = ['brand','model','year','mileage_km','engine_volume','engine_type',
               'transmission','body_type','condition','photos_count',
               'asking_price','text'];
  const car = {};
  ids.forEach(k => { const v = document.getElementById(k).value; if (v !== '') car[k] = v; });
  const out = document.getElementById('out');
  out.className = ''; out.innerHTML = '<div class="card">Calculating…</div>';
  const r = await fetch('/api/estimate', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(car)});
  const d = await r.json();
  if (d.error){ out.innerHTML = '<div class="card note warn">Error: '+esc(d.error)+'</div>'; return; }

  let h = '<div class="card"><div class="muted">Estimated fair listing price</div>'
        + '<div class="big">' + money(d.fair_price) + '</div>'
        + '<div class="range">likely range ' + money(d.range_low)
        + ' — ' + money(d.range_high) + '</div>'
        + '<div class="muted" style="margin-top:10px">The model was trained on '
        + d.trained_rows + ' vehicles and has an average percentage error of about '
        + d.model_mape_pct.toFixed(0) + '%. This estimates a LISTING price, '
        + 'not a guaranteed transaction price.</div></div>';

  if (d.position){
    const p = d.position;
    h += '<div class="card"><h2 style="margin-top:0">Your price among comparable listings</h2>'
      + '<div>' + esc(p.label) + ' — lower than ' + p.percentile.toFixed(0)
      + '% of ' + p.n_similar + ' comparable vehicles</div>'
      + '<div class="muted" style="margin-top:6px">The middle 50% is priced from '
      + money(p.p25) + ' to ' + money(p.p75) + '.</div>'
      + '<div class="note info" style="margin-top:10px">This is a position among '
      + 'advertised prices, not a time-to-sale forecast. The observation history '
      + 'is not long enough to promise a sale date.</div></div>';
  }

  h += '<div class="card"><h2 style="margin-top:0">What drives the estimate</h2>';
  d.drivers.forEach(x => {
    const up = x.effect_pct >= 0;
    const w = Math.min(100, Math.abs(x.effect_pct));
    h += '<div class="row"><span>' + esc(featureLabel(x.feature)) + ' <span class="muted">'
       + esc(valueLabel(x.value)) + '</span></span><b>' + (up?'+':'') + x.effect_pct.toFixed(0)
       + '%</b></div><div class="bar"><i class="' + (up?'up':'down')
       + '" style="width:' + w + '%"></i></div>';
  });
  h += '<div class="muted" style="margin-top:10px">How each characteristic '
     + 'changes this vehicle’s estimate.</div></div>';

  if (d.warnings.length){
    h += '<div class="card"><h2 style="margin-top:0">How to improve the listing</h2>';
    d.warnings.forEach(w => h += '<div class="note warn">' + esc(w) + '</div>');
    h += '</div>';
  }

  if (d.similar.length){
    h += '<div class="card"><h2 style="margin-top:0">Comparable listings</h2>'
       + '<table><tr><th>Vehicle</th><th>Year</th><th>Mileage</th><th>Price</th></tr>';
    d.similar.forEach(s => {
      h += '<tr><td>' + esc(s.brand) + ' ' + esc(s.model) + '</td><td>' + esc(s.year)
         + '</td><td>' + (s.mileage_km ? Math.round(s.mileage_km).toLocaleString('en') : '—')
         + '</td><td>' + money(s.price_tenge) + '</td></tr>';
    });
    h += '</table></div>';
  }
  out.innerHTML = h;
}
</script>"""
    )
    return _page("Vehicle price estimate", body)
