# -*- coding: utf-8 -*-
"""Страница разметки повреждений: фото, рамка мышью, клавиши.

Отдельным файлом от pages.py, потому что это единственная страница проекта
со сложным поведением на стороне браузера — рисование рамки поверх картинки.
Держать её вместе с формой оценки значило бы смешать двести строк JavaScript
с тремя полями ввода.

ЧТО ВАЖНО ДЛЯ СКОРОСТИ РАЗМЕТКИ. Триста кадров по сорок секунд — это три
часа, и разница между удобным и неудобным интерфейсом тут решает, будет
разметка сделана или нет. Поэтому:

  клавиши      D — повреждение, I — целая, U — не понять, стрелки — навигация
  без мыши     «целая» и «не понять» ставятся одной клавишей, рамка не нужна
  автопереход  после сохранения сразу следующий кадр
  возврат      стрелка влево возвращает к предыдущему, метку можно поправить

Картинки отдаются локально (маршрут /photos), к kolesa.kz не идёт ни одного
запроса — тот же принцип, что в карточках вердиктов: ручной браузинг по
сайту однажды уже помог положить IP.
"""

from __future__ import annotations

import html
import json

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Разметка повреждений</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, -apple-system, sans-serif;
         margin: 0; padding: 16px; background: #0f1115; color: #e6e8ec; }
  header { display: flex; align-items: baseline; gap: 18px; margin-bottom: 12px; }
  h1 { font-size: 17px; margin: 0; font-weight: 600; }
  .muted { color: #8b93a1; font-size: 13px; }
  #stage { position: relative; display: inline-block; max-width: 100%;
           border-radius: 8px; overflow: hidden; background: #1a1d24; }
  #shot { display: block; max-width: min(1100px, 92vw); max-height: 74vh;
          user-select: none; -webkit-user-drag: none; }
  #box { position: absolute; border: 2px solid #ff5f56; background: rgba(255,95,86,.15);
         pointer-events: none; display: none; border-radius: 2px; }
  .row { margin-top: 12px; display: flex; gap: 10px; align-items: center;
         flex-wrap: wrap; }
  button { font: inherit; padding: 7px 14px; border-radius: 7px; cursor: pointer;
           border: 1px solid #333a47; background: #1a1d24; color: #e6e8ec; }
  button:hover { background: #232833; }
  button.on { border-color: #4f8cff; background: #16233a; }
  kbd { font: 12px ui-monospace, monospace; background: #232833; color: #b9c0cc;
        border: 1px solid #333a47; border-radius: 4px; padding: 1px 5px; }
  #hint { color: #8b93a1; font-size: 13px; min-height: 20px; }
  #meta { color: #8b93a1; font-size: 13px; }
  input[type=text] { font: inherit; padding: 6px 9px; border-radius: 6px;
    border: 1px solid #333a47; background: #14171d; color: #e6e8ec; width: 320px; }
  .ok { color: #4ade80; }
  .warn { color: #fbbf24; }
</style>

<header>
  <h1>Разметка повреждений</h1>
  <span class="muted" id="progress"></span>
  <span class="muted" id="counts"></span>
</header>

<div id="stage">
  <img id="shot" alt="фотография объявления">
  <div id="box"></div>
</div>

<div class="row">
  <button id="b-damaged">повреждение <kbd>D</kbd></button>
  <button id="b-intact">целая <kbd>I</kbd></button>
  <button id="b-unclear">не понять <kbd>U</kbd></button>
  <span class="muted">|</span>
  <button id="b-prev"><kbd>←</kbd></button>
  <button id="b-next"><kbd>→</kbd></button>
  <input type="text" id="comment" placeholder="комментарий (необязательно)">
</div>

<div class="row"><span id="hint"></span></div>
<div class="row"><span id="meta"></span></div>

<script>
const QUEUE = __QUEUE__;
let i = 0, box = null, drawing = null, mode = null;

const img = document.getElementById('shot');
const boxEl = document.getElementById('box');
const hint = document.getElementById('hint');

function show() {
  const it = QUEUE[i];
  if (!it) { hint.textContent = 'Очередь закончилась.'; return; }
  img.src = '/photos/' + it.path.replace(/^data\\/photos\\//, '');
  box = null; mode = it.label || null;
  boxEl.style.display = 'none';
  document.getElementById('comment').value = it.comment || '';
  document.getElementById('progress').textContent = (i + 1) + ' из ' + QUEUE.length;
  document.getElementById('meta').textContent =
    it.ad_id + ' · кадр ' + it.position +
    (it.suspect ? ' · отмечено как возможное повреждение' : '') +
    (it.price ? ' · ' + it.price : '');
  paintButtons();
  hint.textContent = mode ? 'уже размечено: ' + mode : '';
  if (it.x1) { drawBoxFromRel(it.x1, it.y1, it.x2, it.y2); }
}

function paintButtons() {
  for (const k of ['damaged', 'intact', 'unclear'])
    document.getElementById('b-' + k).classList.toggle('on', mode === k);
}

function rel(e) {
  const r = img.getBoundingClientRect();
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
           y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
}

function drawBoxFromRel(x1, y1, x2, y2) {
  const r = img.getBoundingClientRect(), s = document.getElementById('stage');
  const o = s.getBoundingClientRect();
  boxEl.style.left = (r.left - o.left + x1 * r.width) + 'px';
  boxEl.style.top = (r.top - o.top + y1 * r.height) + 'px';
  boxEl.style.width = ((x2 - x1) * r.width) + 'px';
  boxEl.style.height = ((y2 - y1) * r.height) + 'px';
  boxEl.style.display = 'block';
  box = [x1, y1, x2, y2];
}

img.addEventListener('mousedown', e => { e.preventDefault(); drawing = rel(e); });
window.addEventListener('mousemove', e => {
  if (!drawing) return;
  const p = rel(e);
  drawBoxFromRel(Math.min(drawing.x, p.x), Math.min(drawing.y, p.y),
                 Math.max(drawing.x, p.x), Math.max(drawing.y, p.y));
});
window.addEventListener('mouseup', () => {
  if (!drawing) return;
  drawing = null;
  if (box && (box[2] - box[0] < 0.01 || box[3] - box[1] < 0.01)) {
    box = null; boxEl.style.display = 'none';
    hint.textContent = 'Рамка слишком мелкая — обведите заметную область.';
    return;
  }
  if (box) { mode = 'damaged'; paintButtons(); save(); }
});

async function save() {
  const it = QUEUE[i];
  if (!mode) { hint.textContent = 'Выберите метку.'; return; }
  if (mode === 'damaged' && !box) {
    hint.textContent = 'Обведите повреждение мышью — метка без рамки не нужна.';
    return;
  }
  const body = { ad_id: it.ad_id, position: it.position, path: it.path,
                 label: mode, box: box,
                 comment: document.getElementById('comment').value };
  const r = await fetch('/damage/label', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) { hint.innerHTML = '<span class="warn">' + (j.error || 'ошибка') + '</span>'; return; }
  it.label = mode; it.comment = body.comment;
  if (box) { it.x1 = box[0]; it.y1 = box[1]; it.x2 = box[2]; it.y2 = box[3]; }
  hint.innerHTML = '<span class="ok">сохранено</span>';
  document.getElementById('counts').textContent =
    'повреждений: ' + j.stats.damaged + ' · целых: ' + j.stats.intact +
    ' · неясных: ' + j.stats.unclear;
  setTimeout(() => { i = Math.min(i + 1, QUEUE.length - 1); show(); }, 180);
}

function pick(m) {
  mode = m; paintButtons();
  if (m === 'damaged') { hint.textContent = 'Обведите повреждение мышью.'; return; }
  box = null; boxEl.style.display = 'none';
  save();
}

document.getElementById('b-damaged').onclick = () => pick('damaged');
document.getElementById('b-intact').onclick = () => pick('intact');
document.getElementById('b-unclear').onclick = () => pick('unclear');
document.getElementById('b-prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('b-next').onclick = () => {
  i = Math.min(i + 1, QUEUE.length - 1); show(); };

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 'd' || k === 'в') pick('damaged');
  else if (k === 'i' || k === 'ш') pick('intact');
  else if (k === 'u' || k === 'г') pick('unclear');
  else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); }
  else if (e.key === 'ArrowRight') { i = Math.min(i + 1, QUEUE.length - 1); show(); }
});

show();
</script>
"""


def page(queue_rows: list[dict], counts: dict) -> str:
    """HTML страницы. Очередь уезжает в JavaScript одним куском.

    json.dumps со всеми полями сразу, а не подстановка по одному: очередь
    может быть на четыреста кадров, и генерировать четыреста блоков разметки
    ради того, чтобы браузер показывал по одному, бессмысленно.
    """
    q = json.dumps(queue_rows, ensure_ascii=False)
    out = TEMPLATE.replace("__QUEUE__", q)
    tail = (f"повреждений: {counts.get('damaged', 0)} · "
            f"целых: {counts.get('intact', 0)} · "
            f"неясных: {counts.get('unclear', 0)}")
    return out.replace('id="counts"></span>',
                       f'id="counts">{html.escape(tail)}</span>')
