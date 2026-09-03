# -*- coding: utf-8 -*-
"""Damage labelling page with centered photos, boxes, and confirmation.

This page is separate from ``pages.py`` because it is the only interface with
substantial browser-side behaviour: drawing bounding boxes over an image.

A box is never saved automatically. The first implementation wrote a
``damaged`` label as soon as the mouse was released, although an annotator may
still want to inspect, redraw, or cancel it. The journal is manual ground truth,
so the workflow is now **draw → choose a label → confirm**. Multiple damage
areas can be stored as separate boxes on one frame.

Keyboard shortcuts keep a several-hundred-frame review practical: D/W/P/I/U
select a label, Enter confirms, Escape cancels, and arrow keys navigate.

Images are served locally through ``/photos``. Labelling never requests
kolesa.kz, which avoids adding accidental traffic during manual review.
"""

from __future__ import annotations

import html
import json

from kz.report.photo_labels import LABELS, MAX_BOXES_PER_FRAME

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Damage labeling</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0e1014; --panel: #161a21; --line: #262c37;
    --text: #e8eaef; --muted: #8b93a1; --accent: #4f8cff; --bad: #ff5f56;
    --ok: #4ade80; --warn: #fbbf24;
  }
  * { box-sizing: border-box }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         min-height: 100vh; display: flex; flex-direction: column;
         align-items: center; padding: 20px 16px 40px; }

  header { width: min(1100px, 100%); display: flex; align-items: center;
           gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
  h1 { font-size: 17px; margin: 0; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--muted); font-size: 13px }
  .sub a { color: var(--accent); text-decoration: none }
  .spacer { flex: 1 }
  .pill { background: var(--panel); border: 1px solid var(--line);
          border-radius: 999px; padding: 4px 12px; font-size: 13px;
          color: var(--muted); white-space: nowrap; }
  .pill b { color: var(--text); font-weight: 600 }

  #barwrap { width: min(1100px, 100%); height: 3px; background: var(--line);
             border-radius: 2px; overflow: hidden; margin-bottom: 18px; }
  #bar { height: 100%; width: 0; background: var(--accent); transition: width .2s }

  main { display: flex; flex-direction: column; align-items: center;
         gap: 14px; width: 100%; }
  #stage { position: relative; line-height: 0; border-radius: 10px;
           overflow: hidden; background: var(--panel);
           border: 1px solid var(--line); cursor: crosshair; }
  #shot { display: block; max-width: min(1100px, 92vw); max-height: 68vh;
          user-select: none; -webkit-user-drag: none; }
  #box { position: absolute; border: 2px solid var(--bad); border-radius: 2px;
         background: rgba(255,95,86,.12); pointer-events: none; display: none;
         box-shadow: 0 0 0 9999px rgba(0,0,0,.35); }
  .saved-box { position: absolute; border: 2px solid var(--warn);
               border-radius: 2px; background: rgba(251,191,36,.10);
               color: #111; font: 700 11px/18px system-ui; text-align: center;
               pointer-events: none; min-width: 18px; min-height: 18px; }

  #meta { color: var(--muted); font-size: 13px; text-align: center;
          min-height: 20px; }
  #meta .flag { color: var(--warn) }

  .row { display: flex; gap: 10px; align-items: center; justify-content: center;
         flex-wrap: wrap; }
  button { font: inherit; padding: 9px 16px; border-radius: 8px;
           cursor: pointer; border: 1px solid var(--line);
           background: var(--panel); color: var(--text); transition: .12s; }
  button:hover { border-color: #3a4353; background: #1c212a }
  button.primary { background: var(--accent); border-color: var(--accent);
                   color: #08101f; font-weight: 600 }
  button.primary:hover { filter: brightness(1.08) }
  button.sel { border-color: var(--accent); background: #15243c }
  button:disabled { opacity: .4; cursor: default }
  kbd { font: 12px ui-monospace, SFMono-Regular, monospace; background: #232833;
        color: #b9c0cc; border: 1px solid var(--line); border-radius: 4px;
        padding: 1px 5px; margin-left: 6px; }

  /* Confirmation appears after drawing; the UI never decides for the annotator. */
  #ask { display: none; width: min(560px, 92vw); background: var(--panel);
         border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px;
         box-shadow: 0 12px 40px rgba(0,0,0,.45); }
  #ask.on { display: block }
  #ask h3 { margin: 0 0 4px; font-size: 15px; font-weight: 600 }
  #ask p { margin: 0 0 12px; color: var(--muted); font-size: 13px }
  #ask .row { justify-content: flex-start }
  #comment { font: inherit; padding: 8px 10px; border-radius: 7px;
             border: 1px solid var(--line); background: #11141a;
             color: var(--text); width: 100%; }
  #commentrow { max-width: 640px; margin: 10px auto 0 }
  #hint { font-size: 13px; min-height: 20px; text-align: center }
  .pill.tap { cursor: pointer; user-select: none }
  .pill.tap:hover { border-color: var(--accent); color: var(--text) }
  .pill.on { border-color: var(--accent); color: var(--accent) }
  .hid { display: none }
  #again { text-align: center; font-size: 13px; margin-top: 8px;
           color: var(--warn) }
  #legend { max-width: 640px; margin: 10px auto 0; text-align: center;
            line-height: 1.5 }
  .ok { color: var(--ok) } .warn { color: var(--warn) }
</style>

<header>
  <h1>Damage labeling</h1>
  <span class="sub"><a href="/">← Home</a></span>
  <span class="spacer"></span>
  <span class="pill">Frame <b id="pos">—</b></span>
  <span class="pill tap" data-lab="damaged">Damaged <b id="c-damaged">0</b></span>
  <span class="pill tap" data-lab="wreck">Wreck <b id="c-wreck">0</b></span>
  <span class="pill tap" data-lab="parts">Parts <b id="c-parts">0</b></span>
  <span class="pill tap" data-lab="intact">Intact <b id="c-intact">0</b></span>
  <span class="pill tap" data-lab="unclear">Unclear <b id="c-unclear">0</b></span>
  <span class="pill">Needs review <b id="c-needs-review">0</b></span>
  <span class="pill hid" id="backpill">← Back to queue<kbd>Esc</kbd></span>
</header>

<div id="barwrap"><div id="bar"></div></div>

<main>
  <div id="stage">
    <img id="shot" alt="vehicle listing photo">
    <div id="saved-boxes"></div>
    <div id="box"></div>
  </div>

  <div id="meta"></div>
  <div id="again"></div>

  <div class="row" id="quick">
    <button id="b-intact">Intact<kbd>I</kbd></button>
    <button id="b-wreck">Wreck<kbd>W</kbd></button>
    <button id="b-parts">Parts<kbd>P</kbd></button>
    <button id="b-unclear">Unclear<kbd>U</kbd></button>
    <button id="b-pop-box" disabled>Remove last box</button>
    <span class="sub">Boxes: <b id="boxcount">0</b></span>
    <span class="sub">or draw a box around an impact/dent</span>
  </div>

  <div class="row" id="commentrow">
    <input type="text" id="comment"
           placeholder="Optional note: rust, filler, or anything unusual">
  </div>

  <p class="sub" id="legend">
    <b>Main rule:</b> poor appearance is not the same as impact damage. Look
    for geometric deformation: a dent, crease, broken panel, or displaced part.<br>
    <b>Intact = no impact/dent.</b> Rust, dirt, and scuffs still belong here.
    Record visible rust in the note so that signal is not lost.<br>
    <b>Damaged versus Wreck:</b> if one local area can be boxed, choose Damaged.
    If an entire front or rear assembly is destroyed and no useful local box
    exists, choose Wreck.<br>
    <b>Boxes:</b> draw each separate impact area separately. After the first,
    select Add another box, draw the next one, then save the frame. Keep road
    and sky outside the box. Boxes are retained with any label, including an
    Intact frame where a rust area was marked for future analysis.
  </p>

  <div id="ask">
    <h3>What is inside the box?</h3>
    <p>You can redraw the box; nothing is written until Save is pressed.
       Damaged means one local impact, dent, crease, or broken panel. Choose
       Wreck when the whole assembly is destroyed and a local box is not
       meaningful. Choose Parts when an engine, transmission, or other major
       component has been removed; for the latter two labels, the whole frame
       is the evidence.</p>
    <div class="row">
      <button id="a-damaged" class="sel">Damaged<kbd>D</kbd></button>
      <button id="a-wreck">Wreck<kbd>W</kbd></button>
      <button id="a-parts">Parts<kbd>P</kbd></button>
      <button id="a-unclear">Unclear<kbd>U</kbd></button>
      <button id="a-intact">Intact<kbd>I</kbd></button>
    </div>
    <div class="row">
      <button id="a-add">Add another box</button>
      <button id="a-save" class="primary">Save<kbd>Enter</kbd></button>
      <button id="a-cancel">Cancel box<kbd>Esc</kbd></button>
    </div>
  </div>

  <div id="hint"></div>

  <div class="row">
    <button id="b-prev">← Previous</button>
    <button id="b-next">Next →</button>
  </div>
</main>

<script>
const QUEUE = __QUEUE__;
const DONE = __DONE__;
const MAX_BOXES = __MAX_BOXES__;
// Completed-frame review is a second mode of this page. Keeping it here lets
// an annotator correct labels without interrupting the labelling flow.
let view = QUEUE, mode = 'queue';
let i = 0, box = null, boxes = [], drawing = null, choice = null;

const img = document.getElementById('shot');
const boxEl = document.getElementById('box');
const savedBoxesEl = document.getElementById('saved-boxes');
const ask = document.getElementById('ask');
const hint = document.getElementById('hint');

function show() {
  const it = view[i];
  if (!it) { hint.textContent = 'The queue is complete.'; return; }
  box = null; choice = null;
  boxes = Array.isArray(it.boxes) ? it.boxes.map(b => b.map(Number)) : [];
  if (!boxes.length && it.x1 !== undefined && it.x1 !== null && it.x1 !== '')
    boxes = [[+it.x1, +it.y1, +it.x2, +it.y2]];
  boxEl.style.display = 'none';
  renderBoxes();
  closeAsk();
  img.src = '/photos/' + it.path.replace(/^data\\/photos\\//, '');
  document.getElementById('comment').value = it.comment || '';
  document.getElementById('pos').textContent = (i + 1) + ' / ' + view.length;
  document.getElementById('bar').style.width =
    ((i + 1) / view.length * 100) + '%';
  document.getElementById('meta').innerHTML =
    it.ad_id + ' · frame ' + it.position +
    (it.price ? ' · ' + it.price : '') +
    (it.suspect ? ' · <span class="flag">listing flagged as potentially damaged</span>' : '');
  const again = document.getElementById('again');
  again.innerHTML = it.label
    ? 'Already labeled as ' + LABEL_NAMES[it.label]
      + (it.comment ? ' · ' + it.comment : '')
      + (it.review_status === 'needs_review'
          ? ' · <span class="warn">Needs review: legacy label excluded from CV</span>' : '')
      + ' — edit it here to update the existing journal row'
    : '';
  hint.innerHTML = '';
}

function rel(e) {
  const r = img.getBoundingClientRect();
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
           y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
}

function placeBox(el, coords) {
  const [x1, y1, x2, y2] = coords;
  const r = img.getBoundingClientRect();
  const o = document.getElementById('stage').getBoundingClientRect();
  el.style.left = (r.left - o.left + x1 * r.width) + 'px';
  el.style.top = (r.top - o.top + y1 * r.height) + 'px';
  el.style.width = ((x2 - x1) * r.width) + 'px';
  el.style.height = ((y2 - y1) * r.height) + 'px';
}

function updateBoxCount() {
  document.getElementById('boxcount').textContent = boxes.length;
  document.getElementById('b-pop-box').disabled = boxes.length === 0;
}

function renderBoxes() {
  savedBoxesEl.innerHTML = '';
  boxes.forEach((coords, n) => {
    const el = document.createElement('div');
    el.className = 'saved-box'; el.textContent = n + 1;
    placeBox(el, coords); savedBoxesEl.appendChild(el);
  });
  updateBoxCount();
}

function drawBox(x1, y1, x2, y2) {
  placeBox(boxEl, [x1, y1, x2, y2]);
  boxEl.style.display = 'block';
  box = [x1, y1, x2, y2];
}

img.addEventListener('load', () => {
  renderBoxes();
  if (box) placeBox(boxEl, box);
});
window.addEventListener('resize', () => {
  renderBoxes();
  if (box) placeBox(boxEl, box);
});

function openAsk(pick) {
  choice = pick || 'damaged';
  paintChoice();
  ask.classList.add('on');
  ask.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function closeAsk() { ask.classList.remove('on'); }

function paintChoice() {
  for (const k of ['damaged', 'wreck', 'parts', 'intact', 'unclear'])
    document.getElementById('a-' + k).classList.toggle('sel', choice === k);
}

img.addEventListener('mousedown', e => {
  e.preventDefault();
  if (boxes.length >= MAX_BOXES) {
    hint.innerHTML = '<span class="warn">A frame can contain at most ' + MAX_BOXES + ' boxes.</span>';
    return;
  }
  drawing = rel(e);
});
window.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = rel(e);
  drawBox(Math.min(drawing.x, p.x), Math.min(drawing.y, p.y),
          Math.max(drawing.x, p.x), Math.max(drawing.y, p.y));
});
window.addEventListener('mouseup', () => {
  if (!drawing) return;
  drawing = null;
  if (box && (box[2] - box[0] < 0.01 || box[3] - box[1] < 0.01)) {
    box = null; boxEl.style.display = 'none';
    hint.innerHTML = '<span class="warn">The box is too small; draw a meaningful area.</span>';
    return;
  }
  if (box) { hint.textContent = ''; openAsk('damaged'); }
});

async function commit(label, useBox) {
  // `view` can be the queue or a filtered list of completed frames. Using
  // QUEUE[i] in edit mode once targeted a different frame and risked corrupting
  // the manual journal.
  const it = view[i];
  const finalBoxes = boxes.map(b => b.slice());
  if (useBox && box) finalBoxes.push(box.slice());
  const body = { ad_id: it.ad_id, position: it.position, path: it.path,
                 label: label, boxes: finalBoxes,
                 comment: document.getElementById('comment').value };
  const r = await fetch('/damage/label', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) {
    hint.innerHTML = '<span class="warn">' + (j.error || 'Error') + '</span>';
    return;
  }
  boxes = finalBoxes;
  it.label = label; it.comment = body.comment; it.boxes = finalBoxes;
  // Keep DONE synchronized with the journal so review does not show stale data.
  const k = DONE.findIndex(r => r.ad_id === it.ad_id && r.position === it.position);
  const rec = { ad_id: it.ad_id, position: it.position, path: it.path,
                label: label, comment: body.comment, boxes: finalBoxes };
  for (const target of [it, rec]) {
    for (const key of ['x1', 'y1', 'x2', 'y2']) delete target[key];
    if (finalBoxes.length)
      [target.x1, target.y1, target.x2, target.y2] = finalBoxes[0];
  }
  if (k >= 0) DONE[k] = rec; else DONE.push(rec);
  for (const k of ['damaged', 'wreck', 'parts', 'intact', 'unclear'])
    document.getElementById('c-' + k).textContent = j.stats[k];
  hint.innerHTML = '<span class="ok">Saved</span>';
  closeAsk();
  setTimeout(() => { if (i < view.length - 1) { i++; show(); } }, 170);
}

// Preserve a drawn box with every label. Older code discarded boxes for
// non-damaged labels and silently lost regions marking rust.
document.getElementById('a-save').onclick = () => commit(choice, !!box);
document.getElementById('a-add').onclick = () => {
  if (!box) {
    hint.innerHTML = '<span class="warn">Draw a box first.</span>';
    return;
  }
  if (boxes.length >= MAX_BOXES) {
    hint.innerHTML = '<span class="warn">A frame can contain at most ' + MAX_BOXES + ' boxes.</span>';
    return;
  }
  boxes.push(box.slice()); box = null; boxEl.style.display = 'none';
  renderBoxes(); closeAsk();
  hint.innerHTML = '<span class="ok">Box added. Draw another one or save the label.</span>';
};
document.getElementById('a-cancel').onclick = () => {
  box = null; boxEl.style.display = 'none'; closeAsk(); hint.textContent = '';
};
document.getElementById('a-damaged').onclick = () => { choice = 'damaged'; paintChoice(); };
document.getElementById('a-parts').onclick = () => { choice = 'parts'; paintChoice(); };
document.getElementById('a-intact').onclick = () => { choice = 'intact'; paintChoice(); };
document.getElementById('a-unclear').onclick = () => { choice = 'unclear'; paintChoice(); };

/* Labels without local evidence can be saved with one button and no box. */
document.getElementById('b-intact').onclick = () => commit('intact', false);
document.getElementById('b-parts').onclick = () => commit('parts', false);
document.getElementById('b-unclear').onclick = () => commit('unclear', false);
document.getElementById('b-wreck').onclick = () => commit('wreck', false);
document.getElementById('b-pop-box').onclick = () => {
  if (!boxes.length) return;
  boxes.pop(); renderBoxes();
  hint.innerHTML = '<span class="warn">Last box removed. Save the label to commit the change.</span>';
};
document.getElementById('a-wreck').onclick = () => { choice = 'wreck'; paintChoice(); };
document.getElementById('b-prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('b-next').onclick = () => {
  i = Math.min(i + 1, view.length - 1); show(); };

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') {
    // Enter in the note confirms only while the dialog is open. Outside the
    // dialog no label has been selected, so there is nothing to save.
    if (e.key === 'Enter' && ask.classList.contains('on'))
      document.getElementById('a-save').click();
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  const k = e.key.toLowerCase();
  const open = ask.classList.contains('on');
  if (e.key === 'Escape') {
    if (ask.classList.contains('on')) document.getElementById('a-cancel').click();
    else if (mode !== 'queue') setMode(null);
  }
  else if (e.key === 'Enter' && open) { document.getElementById('a-save').click(); }
  else if (k === 'd') {
    if (open) { choice = 'damaged'; paintChoice(); }
    else if (boxes.length) openAsk('damaged');
    else hint.innerHTML = '<span class="warn">Draw a box around the damage first.</span>';
  }
  else if (k === 'w') { open ? (choice = 'wreck', paintChoice())
                                          : commit('wreck', false); }
  else if (k === 'p') { open ? (choice = 'parts', paintChoice())
                                          : commit('parts', false); }
  else if (k === 'i') { open ? (choice = 'intact', paintChoice())
                                          : commit('intact', false); }
  else if (k === 'u') { open ? (choice = 'unclear', paintChoice())
                                          : commit('unclear', false); }
  else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); }
  else if (e.key === 'ArrowRight') { i = Math.min(i + 1, view.length - 1); show(); }
});

const LABEL_NAMES = { damaged: 'Damaged', wreck: 'Wreck', parts: 'Parts',
                      intact: 'Intact', unclear: 'Unclear' };

function setMode(label) {
  const back = document.getElementById('backpill');
  document.querySelectorAll('.pill.tap').forEach(
    p => p.classList.toggle('on', p.dataset.lab === label));
  if (!label) {
    mode = 'queue'; view = QUEUE; back.classList.add('hid');
  } else {
    const rows = DONE.filter(r => r.label === label);
    if (!rows.length) {
      hint.innerHTML = '<span class="warn">No frames have this label yet.</span>';
      return;
    }
    mode = label; view = rows; back.classList.remove('hid');
  }
  i = 0; show();
}

document.querySelectorAll('.pill.tap').forEach(p => {
  p.onclick = () => setMode(mode === p.dataset.lab ? null : p.dataset.lab);
});
document.getElementById('backpill').onclick = () => setMode(null);

show();
</script>
"""


def page(queue_rows: list[dict], counts: dict, done_rows: list[dict] | None = None) -> str:
    """Build the page and serialize the queue into one JavaScript payload.

    A queue may contain hundreds of frames. Serializing one array is simpler
    and smaller than generating hundreds of HTML blocks when only one is shown.
    """
    out = TEMPLATE.replace("__QUEUE__", json.dumps(queue_rows, ensure_ascii=False))
    out = out.replace("__DONE__", json.dumps(done_rows or [], ensure_ascii=False))
    out = out.replace("__MAX_BOXES__", str(MAX_BOXES_PER_FRAME))
    # Derive counters from LABELS. A former hard-coded tuple missed the newly
    # added `parts` label and displayed an incorrect zero count.
    for key in LABELS:
        out = out.replace(
            f'id="c-{key}">0<', f'id="c-{key}">{html.escape(str(counts.get(key, 0)))}<'
        )
    out = out.replace(
        'id="c-needs-review">0<',
        f'id="c-needs-review">{html.escape(str(counts.get("needs_review", 0)))}<',
    )
    return out
