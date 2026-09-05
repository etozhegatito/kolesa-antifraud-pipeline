# -*- coding: utf-8 -*-
"""Browser UI for a blinded, listing-level review of the cheap segment."""

from __future__ import annotations

import html
import json

from kz.report.price_review import (
    DATA_ISSUES,
    EVIDENCE_SOURCES,
    PRICE_VALIDITY,
    VEHICLE_STATES,
)


def _json(value) -> str:
    """Serialize an inline payload without allowing a value to close script."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _buttons(values: dict[str, str], group: str) -> str:
    return "".join(
        f'<button type="button" class="choice" data-group="{group}" '
        f'data-value="{html.escape(key)}" title="{html.escape(description)}">'
        f"{html.escape(key.replace('_', ' ').title())}</button>"
        for key, description in values.items()
    )


TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Below-5M condition review</title>
<style>
:root{color-scheme:light dark;--bg:#0c0f16;--panel:#131824;--line:#273044;
 --text:#e9edf5;--muted:#929bad;--accent:#71a3ff;--ok:#57d38c;--warn:#ffc565}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1260px;margin:auto;padding:22px 20px 70px}header{display:flex;gap:12px;
 align-items:center;flex-wrap:wrap;margin-bottom:14px}h1{font-size:20px;margin:0}.spacer{flex:1}
a{color:var(--accent);text-decoration:none}.pill{border:1px solid var(--line);border-radius:999px;
 padding:5px 12px;color:var(--muted);cursor:pointer;background:var(--panel)}.pill.on{color:var(--accent);border-color:var(--accent)}
.bar{height:3px;background:var(--line);margin-bottom:18px}.bar i{display:block;height:100%;background:var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:18px}
.layout{display:grid;grid-template-columns:minmax(420px,1.25fr) minmax(350px,.75fr);gap:18px}
.hero{height:530px;display:flex;align-items:center;justify-content:center;background:#080a0f;border-radius:11px;overflow:hidden}
.hero img{max-width:100%;max-height:100%;display:block}.thumbs{display:flex;gap:8px;overflow:auto;margin-top:9px}
.thumb{width:82px;height:62px;padding:0;border:2px solid transparent;border-radius:8px;overflow:hidden;background:#080a0f;flex:none}
.thumb.on{border-color:var(--accent)}.thumb img{width:100%;height:100%;object-fit:cover}.muted{color:var(--muted);font-size:13px}
.facts{display:grid;grid-template-columns:1fr 1fr;gap:0 16px;margin:12px 0}.fact{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:7px 0}.fact span:first-child{color:var(--muted)}
.text{white-space:pre-wrap;background:var(--bg);border-radius:9px;padding:11px;margin-top:10px;max-height:180px;overflow:auto}
h2{font-size:15px;margin:16px 0 7px}.choices{display:flex;gap:7px;flex-wrap:wrap}.choice{border:1px solid var(--line);background:var(--bg);color:var(--text);padding:8px 10px;border-radius:8px;cursor:pointer}
.choice.on{border-color:var(--accent);background:#162642;color:#bcd3ff}select,textarea{width:100%;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--text);padding:9px;font:inherit}
textarea{min-height:70px;resize:vertical}.save{width:100%;border:0;border-radius:9px;padding:11px;margin-top:13px;background:var(--accent);color:#07101e;font-weight:700;cursor:pointer}
.save:disabled{opacity:.45;cursor:not-allowed}.nav{display:flex;gap:8px;margin-top:12px}.nav button,.cv{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}.cv{display:inline-block;margin-top:10px}
.notice{border:1px solid #655021;background:#211b0f;color:var(--warn);border-radius:10px;padding:10px 12px;margin-bottom:14px}.ok{color:var(--ok)}
details{border:1px solid var(--line);border-radius:9px;padding:8px 10px;margin-top:12px}details p{margin:7px 0;color:var(--muted);font-size:13px}
@media(max-width:850px){.layout{grid-template-columns:1fr}.hero{height:55vh;min-height:320px}.facts{grid-template-columns:1fr}}
</style>
<div class="wrap">
<header><h1>Below-5M condition review</h1><a href="/">← Home</a><span class="spacer"></span>
 <button class="pill on" id="tab-queue">Queue <b id="queue-count">__QUEUE_COUNT__</b></button>
 <button class="pill" id="tab-reviewed">Reviewed <b id="reviewed-count">__REVIEWED_COUNT__</b></button>
</header>
<div class="bar"><i id="progress"></i></div>
<div class="notice">Label objective evidence, not whether the model was “right”. Model predictions and errors are hidden to prevent anchoring. These labels do not enter price training automatically. The pilot covers the locally downloaded-photo pool, not yet every below-5M listing.</div>
<div id="empty" class="card" hidden>The current view is complete.</div>
<div id="work" class="layout">
 <section class="card">
  <div class="hero"><img id="hero" alt="locally stored listing photo"></div>
  <div class="thumbs" id="thumbs"></div>
  <a class="cv" id="cv-link" href="#">Precisely label this frame for CV →</a>
  <div class="muted">The precise tool stores a frame label and bounding boxes in the separate CV journal.</div>
 </section>
 <section class="card">
  <div><b id="title"></b> <span class="muted" id="position"></span></div>
  <div class="facts" id="facts"></div>
  <div class="text" id="description"></div>
  <div class="text" id="seller"></div>
  <details><summary>Exact labeling rules</summary>
   <p><b>Normal</b> means no material issue is visible or disclosed; an old-looking car can still be normal.</p>
   <p><b>Cosmetic</b> includes rust, scratches, paint wear, and scuffs. <b>Repair needed</b> means local impact/dent or a disclosed mechanical repair. <b>Non-running</b>, <b>Wreck</b>, and <b>Parts</b> are kept separate.</p>
   <p>Price meaning is a different question: a damaged but honestly priced vehicle may still have a comparable cash amount. Evidence source says where your state decision came from.</p>
  </details>

  <h2>1. Overall vehicle state</h2><div class="choices">__STATE_BUTTONS__</div>
  <h2>2. What does the advertised amount mean?</h2><div class="choices">__PRICE_BUTTONS__</div>
  <h2>3. Where is the condition evidence?</h2><div class="choices">__EVIDENCE_BUTTONS__</div>
  <h2>Optional data-quality issue</h2><select id="data-issue">__DATA_OPTIONS__</select>
  <h2>Optional note</h2><textarea id="comment" placeholder="What exactly is visible or disclosed?"></textarea>
  <button class="save" id="save" disabled>Save review</button>
  <div id="hint" class="muted"></div>
  <div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div>
 </section>
</div>
</div>
<script>
const QUEUE=__QUEUE__, DONE=__DONE__;
let view=QUEUE, index=0, photoIndex=0, picked={vehicle_state:null,price_validity:null,evidence_source:null};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=v=>v==null?'—':(Number(v)/1e6).toFixed(2).replace(/0+$/,'').replace(/\.$/,'')+'M ₸';
const work=document.getElementById('work'), empty=document.getElementById('empty'), hint=document.getElementById('hint');

function choose(group,value){picked[group]=value;document.querySelectorAll('[data-group="'+group+'"]').forEach(b=>b.classList.toggle('on',b.dataset.value===value));validate();}
document.querySelectorAll('.choice').forEach(b=>b.onclick=()=>choose(b.dataset.group,b.dataset.value));
function validate(){document.getElementById('save').disabled=!picked.vehicle_state||!picked.price_validity||!picked.evidence_source;}
function fact(label,value){return '<div class="fact"><span>'+esc(label)+'</span><b>'+esc(value??'—')+'</b></div>';}
function showPhoto(){const item=view[index],p=item.photos[photoIndex];if(!p)return;document.getElementById('hero').src=p.src;document.querySelectorAll('.thumb').forEach((b,i)=>b.classList.toggle('on',i===photoIndex));document.getElementById('cv-link').href='/damage?ad_id='+encodeURIComponent(item.ad_id)+'&position='+p.position;}
function show(){
 const item=view[index];
 if(!item){work.hidden=true;empty.hidden=false;document.getElementById('progress').style.width='100%';return;}
 work.hidden=false;empty.hidden=true;photoIndex=0;hint.textContent='';
 picked={vehicle_state:item.vehicle_state||null,price_validity:item.price_validity||null,evidence_source:item.evidence_source||null};
 document.querySelectorAll('.choice').forEach(b=>b.classList.toggle('on',picked[b.dataset.group]===b.dataset.value));
 document.getElementById('data-issue').value=item.data_issue||'none';document.getElementById('comment').value=item.comment||'';validate();
 document.getElementById('title').textContent=(item.brand||'')+' '+(item.model||'');document.getElementById('position').textContent='#'+item.ad_id+' · '+(index+1)+' / '+view.length;
 document.getElementById('facts').innerHTML=fact('Price',money(item.price_tenge))+fact('Year',item.year)+fact('Age',item.age)+fact('Mileage',item.mileage_km)+fact('Engine',item.engine_volume)+fact('Transmission',item.transmission)+fact('Body',item.body_type)+fact('Local frames',item.photos.length+' / '+(item.photos_count??'unknown'))+fact('Site condition badge',item.page_status_badge)+fact('Parsed damage terms',item.damage_keywords)+fact('Parsed price basis',item.price_basis);
 document.getElementById('description').innerHTML='<span class="muted">Listing summary</span><br>'+esc(item.description||'No summary');
 document.getElementById('seller').innerHTML='<span class="muted">Seller/detail text</span><br>'+esc(item.seller_comment||item.text_full||'No enriched seller text');
 const thumbs=document.getElementById('thumbs');thumbs.innerHTML='';item.photos.forEach((p,n)=>{const b=document.createElement('button');b.className='thumb';b.innerHTML='<img loading="lazy" src="'+esc(p.src)+'" alt="frame '+p.position+'">';b.onclick=()=>{photoIndex=n;showPhoto()};thumbs.appendChild(b)});showPhoto();
 document.getElementById('progress').style.width=((index+1)/Math.max(1,view.length)*100)+'%';
}
document.getElementById('save').onclick=async()=>{
 const item=view[index],body={ad_id:item.ad_id,...picked,data_issue:document.getElementById('data-issue').value,comment:document.getElementById('comment').value};
 const response=await fetch('/price-review/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),data=await response.json();
 if(!response.ok){hint.textContent=data.error||'Save failed';return;}
 Object.assign(item,body);const old=DONE.findIndex(r=>r.ad_id===item.ad_id);if(old>=0)DONE[old]=item;else DONE.push(item);
 if(view===QUEUE){QUEUE.splice(index,1);if(index>=QUEUE.length)index=Math.max(0,QUEUE.length-1)}
 document.getElementById('queue-count').textContent=QUEUE.length;document.getElementById('reviewed-count').textContent=DONE.length;hint.innerHTML='<span class="ok">Saved</span>';setTimeout(show,120);
};
document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);show()};document.getElementById('next').onclick=()=>{index=Math.min(view.length-1,index+1);show()};
function tab(which){view=which==='queue'?QUEUE:DONE;index=0;document.getElementById('tab-queue').classList.toggle('on',which==='queue');document.getElementById('tab-reviewed').classList.toggle('on',which==='reviewed');show()}
document.getElementById('tab-queue').onclick=()=>tab('queue');document.getElementById('tab-reviewed').onclick=()=>tab('reviewed');show();
</script>"""


def page(queue_rows: list[dict], done_rows: list[dict]) -> str:
    """Render one-page gallery review with a recoverable history tab."""
    options = "".join(
        f'<option value="{html.escape(key)}">{html.escape(key.replace("_", " ").title())}</option>'
        for key in DATA_ISSUES
    )
    return (
        TEMPLATE.replace("__QUEUE__", _json(queue_rows))
        .replace("__DONE__", _json(done_rows))
        .replace("__QUEUE_COUNT__", str(len(queue_rows)))
        .replace("__REVIEWED_COUNT__", str(len(done_rows)))
        .replace("__STATE_BUTTONS__", _buttons(VEHICLE_STATES, "vehicle_state"))
        .replace("__PRICE_BUTTONS__", _buttons(PRICE_VALIDITY, "price_validity"))
        .replace("__EVIDENCE_BUTTONS__", _buttons(EVIDENCE_SOURCES, "evidence_source"))
        .replace("__DATA_OPTIONS__", options)
    )
