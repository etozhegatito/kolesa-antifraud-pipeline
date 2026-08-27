# -*- coding: utf-8 -*-
"""Страница разметки повреждений: фото по центру, рамка, подтверждение.

Отдельным файлом от pages.py, потому что это единственная страница проекта
со сложным поведением на стороне браузера — рисование рамки поверх картинки.
Держать её вместе с формой оценки значило бы смешать двести строк JavaScript
с тремя полями ввода.

ПОЧЕМУ РАМКА НЕ СОХРАНЯЕТ САМА. Первая версия ставила метку «повреждение»
сразу, как только отпущена мышь. Быстро — и неверно: обвести область можно,
чтобы разглядеть её поближе, чтобы поправить границы, чтобы передумать.
Автоматический вывод за человека превращал каждое движение мыши в запись в
журнал, который нельзя восстановить пересчётом.

Теперь порядок такой: **обвёл → появилось окно с выбором → подтвердил**.
Рамку можно перерисовать сколько угодно, пока не нажато «сохранить».

ЧТО ВАЖНО ДЛЯ СКОРОСТИ. Триста кадров — это часа полтора, и разница между
удобным и неудобным интерфейсом решает, будет разметка сделана или нет:

  без рамки    «целая» и «не понять» ставятся одной клавишей, рисовать не надо
  клавиши      D — повреждение кузова, P — разобрана, I — целая,
               U — не понять, Enter — подтвердить
  Esc          отменить рамку и начать заново
  стрелки      назад и вперёд, метку можно поправить
  автопереход  после сохранения сразу следующий кадр

Картинки отдаются локально (маршрут /photos), к kolesa.kz не идёт ни одного
запроса — тот же принцип, что в карточках вердиктов: ручной браузинг по
сайту однажды уже помог положить IP.
"""

from __future__ import annotations

import html
import json

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка повреждений</title>
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

  /* Окно подтверждения: появляется ПОСЛЕ рамки, вывод за человека не делаем */
  #ask { display: none; width: min(560px, 92vw); background: var(--panel);
         border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px;
         box-shadow: 0 12px 40px rgba(0,0,0,.45); }
  #ask.on { display: block }
  #ask h3 { margin: 0 0 4px; font-size: 15px; font-weight: 600 }
  #ask p { margin: 0 0 12px; color: var(--muted); font-size: 13px }
  #ask .row { justify-content: flex-start }
  #comment { font: inherit; padding: 8px 10px; border-radius: 7px;
             border: 1px solid var(--line); background: #11141a;
             color: var(--text); width: 100%; margin: 12px 0 12px; }
  #hint { font-size: 13px; min-height: 20px; text-align: center }
  .ok { color: var(--ok) } .warn { color: var(--warn) }
</style>

<header>
  <h1>Разметка повреждений</h1>
  <span class="sub"><a href="/">← главная</a></span>
  <span class="spacer"></span>
  <span class="pill">кадр <b id="pos">—</b></span>
  <span class="pill">повреждений <b id="c-damaged">0</b></span>
  <span class="pill">разобрано <b id="c-parts">0</b></span>
  <span class="pill">целых <b id="c-intact">0</b></span>
  <span class="pill">неясных <b id="c-unclear">0</b></span>
</header>

<div id="barwrap"><div id="bar"></div></div>

<main>
  <div id="stage">
    <img id="shot" alt="фотография объявления">
    <div id="box"></div>
  </div>

  <div id="meta"></div>

  <div class="row" id="quick">
    <button id="b-intact">целая<kbd>I</kbd></button>
    <button id="b-parts">разобрана<kbd>P</kbd></button>
    <button id="b-unclear">не понять<kbd>U</kbd></button>
    <span class="sub">или обведите повреждение кузова мышью</span>
  </div>

  <div id="ask">
    <h3>Что на выделенной области?</h3>
    <p>Рамку можно перерисовать — запись произойдёт только по кнопке.
       «Разобрана» — если снят двигатель или коробка: там свидетельство весь
       кадр, а не участок.</p>
    <div class="row">
      <button id="a-damaged" class="sel">повреждение кузова<kbd>D</kbd></button>
      <button id="a-parts">разобрана / снят агрегат<kbd>P</kbd></button>
      <button id="a-unclear">не понять<kbd>U</kbd></button>
      <button id="a-intact">целая, рамка не нужна<kbd>I</kbd></button>
    </div>
    <input type="text" id="comment" placeholder="комментарий (необязательно)">
    <div class="row">
      <button id="a-save" class="primary">сохранить<kbd>Enter</kbd></button>
      <button id="a-cancel">отменить рамку<kbd>Esc</kbd></button>
    </div>
  </div>

  <div id="hint"></div>

  <div class="row">
    <button id="b-prev">← назад</button>
    <button id="b-next">вперёд →</button>
  </div>
</main>

<script>
const QUEUE = __QUEUE__;
let i = 0, box = null, drawing = null, choice = null;

const img = document.getElementById('shot');
const boxEl = document.getElementById('box');
const ask = document.getElementById('ask');
const hint = document.getElementById('hint');

function show() {
  const it = QUEUE[i];
  if (!it) { hint.textContent = 'Очередь закончилась.'; return; }
  img.src = '/photos/' + it.path.replace(/^data\\/photos\\//, '');
  box = null; choice = null;
  boxEl.style.display = 'none';
  closeAsk();
  document.getElementById('comment').value = it.comment || '';
  document.getElementById('pos').textContent = (i + 1) + ' / ' + QUEUE.length;
  document.getElementById('bar').style.width =
    ((i + 1) / QUEUE.length * 100) + '%';
  document.getElementById('meta').innerHTML =
    it.ad_id + ' · кадр ' + it.position +
    (it.price ? ' · ' + it.price : '') +
    (it.suspect ? ' · <span class="flag">объявление отмечено как возможно повреждённое</span>' : '');
  hint.innerHTML = it.label
    ? '<span class="ok">уже размечено: ' + it.label + '</span>' : '';
  if (it.x1 !== undefined && it.x1 !== null && it.x1 !== '')
    drawBox(+it.x1, +it.y1, +it.x2, +it.y2);
}

function rel(e) {
  const r = img.getBoundingClientRect();
  return { x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
           y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)) };
}

function drawBox(x1, y1, x2, y2) {
  const r = img.getBoundingClientRect();
  const o = document.getElementById('stage').getBoundingClientRect();
  boxEl.style.left = (r.left - o.left + x1 * r.width) + 'px';
  boxEl.style.top = (r.top - o.top + y1 * r.height) + 'px';
  boxEl.style.width = ((x2 - x1) * r.width) + 'px';
  boxEl.style.height = ((y2 - y1) * r.height) + 'px';
  boxEl.style.display = 'block';
  box = [x1, y1, x2, y2];
}

function openAsk(pick) {
  choice = pick || 'damaged';
  paintChoice();
  ask.classList.add('on');
  ask.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function closeAsk() { ask.classList.remove('on'); }

function paintChoice() {
  for (const k of ['damaged', 'parts', 'intact', 'unclear'])
    document.getElementById('a-' + k).classList.toggle('sel', choice === k);
}

img.addEventListener('mousedown', e => { e.preventDefault(); drawing = rel(e); });
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
    hint.innerHTML = '<span class="warn">Рамка слишком мелкая — обведите заметную область.</span>';
    return;
  }
  if (box) { hint.textContent = ''; openAsk('damaged'); }
});

async function commit(label, useBox) {
  const it = QUEUE[i];
  const body = { ad_id: it.ad_id, position: it.position, path: it.path,
                 label: label, box: useBox ? box : null,
                 comment: document.getElementById('comment').value };
  const r = await fetch('/damage/label', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) {
    hint.innerHTML = '<span class="warn">' + (j.error || 'ошибка') + '</span>';
    return;
  }
  it.label = label; it.comment = body.comment;
  if (useBox && box) { it.x1 = box[0]; it.y1 = box[1]; it.x2 = box[2]; it.y2 = box[3]; }
  for (const k of ['damaged', 'parts', 'intact', 'unclear'])
    document.getElementById('c-' + k).textContent = j.stats[k];
  hint.innerHTML = '<span class="ok">сохранено</span>';
  closeAsk();
  setTimeout(() => { if (i < QUEUE.length - 1) { i++; show(); } }, 170);
}

document.getElementById('a-save').onclick = () =>
  commit(choice, choice === 'damaged');
document.getElementById('a-cancel').onclick = () => {
  box = null; boxEl.style.display = 'none'; closeAsk(); hint.textContent = '';
};
document.getElementById('a-damaged').onclick = () => { choice = 'damaged'; paintChoice(); };
document.getElementById('a-parts').onclick = () => { choice = 'parts'; paintChoice(); };
document.getElementById('a-intact').onclick = () => { choice = 'intact'; paintChoice(); };
document.getElementById('a-unclear').onclick = () => { choice = 'unclear'; paintChoice(); };

/* Без рамки — одной кнопкой: рисовать, чтобы сказать «целая», незачем. */
document.getElementById('b-intact').onclick = () => commit('intact', false);
document.getElementById('b-parts').onclick = () => commit('parts', false);
document.getElementById('b-unclear').onclick = () => commit('unclear', false);
document.getElementById('b-prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('b-next').onclick = () => {
  i = Math.min(i + 1, QUEUE.length - 1); show(); };

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') {
    if (e.key === 'Enter') document.getElementById('a-save').click();
    return;
  }
  const k = e.key.toLowerCase();
  const open = ask.classList.contains('on');
  if (e.key === 'Escape') { document.getElementById('a-cancel').click(); }
  else if (e.key === 'Enter' && open) { document.getElementById('a-save').click(); }
  else if (k === 'd' || k === 'в') {
    if (open) { choice = 'damaged'; paintChoice(); }
    else hint.innerHTML = '<span class="warn">Сначала обведите повреждение мышью.</span>';
  }
  else if (k === 'p' || k === 'з') { open ? (choice = 'parts', paintChoice())
                                          : commit('parts', false); }
  else if (k === 'i' || k === 'ш') { open ? (choice = 'intact', paintChoice())
                                          : commit('intact', false); }
  else if (k === 'u' || k === 'г') { open ? (choice = 'unclear', paintChoice())
                                          : commit('unclear', false); }
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
    out = TEMPLATE.replace("__QUEUE__", json.dumps(queue_rows, ensure_ascii=False))
    for key in ("damaged", "intact", "unclear"):
        out = out.replace(f'id="c-{key}">0<',
                          f'id="c-{key}">{html.escape(str(counts.get(key, 0)))}<')
    return out
