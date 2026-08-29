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
  клавиши      D — удар/вмятина, W — авария, P — разобрана, I — целая,
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

from kz.report.photo_labels import LABELS

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
  <h1>Разметка повреждений</h1>
  <span class="sub"><a href="/">← главная</a></span>
  <span class="spacer"></span>
  <span class="pill">кадр <b id="pos">—</b></span>
  <span class="pill tap" data-lab="damaged">повреждений <b id="c-damaged">0</b></span>
  <span class="pill tap" data-lab="wreck">аварий <b id="c-wreck">0</b></span>
  <span class="pill tap" data-lab="parts">разобрано <b id="c-parts">0</b></span>
  <span class="pill tap" data-lab="intact">целых <b id="c-intact">0</b></span>
  <span class="pill tap" data-lab="unclear">неясных <b id="c-unclear">0</b></span>
  <span class="pill hid" id="backpill">← вернуться к очереди<kbd>Esc</kbd></span>
</header>

<div id="barwrap"><div id="bar"></div></div>

<main>
  <div id="stage">
    <img id="shot" alt="фотография объявления">
    <div id="box"></div>
  </div>

  <div id="meta"></div>
  <div id="again"></div>

  <div class="row" id="quick">
    <button id="b-intact">целая<kbd>I</kbd></button>
    <button id="b-wreck">авария<kbd>W</kbd></button>
    <button id="b-parts">разобрана<kbd>P</kbd></button>
    <button id="b-unclear">не понять<kbd>U</kbd></button>
    <span class="sub">или обведите удар мышью</span>
  </div>

  <div class="row" id="commentrow">
    <input type="text" id="comment"
           placeholder="комментарий: ржавчина, шпатлёвка, что заметил (необязательно)">
  </div>

  <p class="sub" id="legend">
    <b>целая</b> — ударов и вмятин нет. Ржавчина, грязь, потёртости тоже
    сюда: ржавчину сеть уже различает сама, а вмятину нет — ради неё и
    размечаем. Заметил ржавчину — напиши в комментарий, не теряй.<br>
    <b>рамка или авария</b> — по простому правилу: можно обвести одно
    место, значит «повреждение»; разрушен весь перёд или зад и обводить
    нечего, значит «авария».<br>
    <b>рамка</b> — обводи самое явное место, а не всё подряд. Пропущенный
    кусок ничего не портит: рамка нужна как чистый пример повреждения.
    А рамка наполовину из асфальта и неба портит. Рамка сохраняется при
    любой метке — обвёл ржавчину и поставил «целая», область запишется.
  </p>

  <div id="ask">
    <h3>Что на выделенной области?</h3>
    <p>Рамку можно перерисовать — запись произойдёт только по кнопке.
       «Повреждение» — это удар в одном месте: вмятина, залом, разбитая
       деталь. «Авария» — если разрушен весь узел и обводить нечего.
       «Разобрана» — если снят двигатель или коробка. У последних двух
       свидетельство весь кадр, а не участок.</p>
    <div class="row">
      <button id="a-damaged" class="sel">повреждение кузова<kbd>D</kbd></button>
      <button id="a-wreck">серьёзная авария<kbd>W</kbd></button>
      <button id="a-parts">разобрана / снят агрегат<kbd>P</kbd></button>
      <button id="a-unclear">не понять<kbd>U</kbd></button>
      <button id="a-intact">целая<kbd>I</kbd></button>
    </div>
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
const DONE = __DONE__;
// Просмотр уже размеченного — второй режим той же страницы. Отдельной
// страницы не делаем: смысл в том, чтобы поправить метку не выходя из
// разметки, а переход туда-обратно сбивал бы темп.
let view = QUEUE, mode = 'queue';
let i = 0, box = null, drawing = null, choice = null;

const img = document.getElementById('shot');
const boxEl = document.getElementById('box');
const ask = document.getElementById('ask');
const hint = document.getElementById('hint');

function show() {
  const it = view[i];
  if (!it) { hint.textContent = 'Очередь закончилась.'; return; }
  img.src = '/photos/' + it.path.replace(/^data\\/photos\\//, '');
  box = null; choice = null;
  boxEl.style.display = 'none';
  closeAsk();
  document.getElementById('comment').value = it.comment || '';
  document.getElementById('pos').textContent = (i + 1) + ' / ' + view.length;
  document.getElementById('bar').style.width =
    ((i + 1) / view.length * 100) + '%';
  document.getElementById('meta').innerHTML =
    it.ad_id + ' · кадр ' + it.position +
    (it.price ? ' · ' + it.price : '') +
    (it.suspect ? ' · <span class="flag">объявление отмечено как возможно повреждённое</span>' : '');
  const again = document.getElementById('again');
  again.innerHTML = it.label
    ? 'вы уже отмечали это как «' + RU[it.label] + '»'
      + (it.comment ? ' · ' + it.comment : '')
      + ' — можно поправить, запись обновится на месте'
    : '';
  hint.innerHTML = '';
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
  for (const k of ['damaged', 'wreck', 'parts', 'intact', 'unclear'])
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
  // `view` бывает очередью или отфильтрованным списком уже размеченного.
  // QUEUE[i] в режиме правки указывал на ДРУГОЙ кадр и мог испортить
  // ручной журнал при попытке исправить старую метку.
  const it = view[i];
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
  // держим DONE в согласии с журналом, иначе просмотр покажет старую метку
  const k = DONE.findIndex(r => r.ad_id === it.ad_id && r.position === it.position);
  const rec = { ad_id: it.ad_id, position: it.position, path: it.path,
                label: label, comment: body.comment };
  if (useBox && box) { rec.x1 = box[0]; rec.y1 = box[1]; rec.x2 = box[2]; rec.y2 = box[3]; }
  if (k >= 0) DONE[k] = rec; else DONE.push(rec);
  if (useBox && box) { it.x1 = box[0]; it.y1 = box[1]; it.x2 = box[2]; it.y2 = box[3]; }
  for (const k of ['damaged', 'wreck', 'parts', 'intact', 'unclear'])
    document.getElementById('c-' + k).textContent = j.stats[k];
  hint.innerHTML = '<span class="ok">сохранено</span>';
  closeAsk();
  setTimeout(() => { if (i < view.length - 1) { i++; show(); } }, 170);
}

// Рамку сохраняем при ЛЮБОЙ метке, если она нарисована: раньше её
// отбрасывали для всего кроме «повреждения», и обведённая ржавчина
// пропадала молча.
document.getElementById('a-save').onclick = () => commit(choice, !!box);
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
document.getElementById('b-wreck').onclick = () => commit('wreck', false);
document.getElementById('a-wreck').onclick = () => { choice = 'wreck'; paintChoice(); };
document.getElementById('b-prev').onclick = () => { i = Math.max(0, i - 1); show(); };
document.getElementById('b-next').onclick = () => {
  i = Math.min(i + 1, view.length - 1); show(); };

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') {
    // Enter в комментарии подтверждает только когда диалог открыт: вне его
    // метка ещё не выбрана, и сохранять нечего.
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
  else if (k === 'd' || k === 'в') {
    if (open) { choice = 'damaged'; paintChoice(); }
    else hint.innerHTML = '<span class="warn">Сначала обведите повреждение мышью.</span>';
  }
  else if (k === 'w' || k === 'ц') { open ? (choice = 'wreck', paintChoice())
                                          : commit('wreck', false); }
  else if (k === 'p' || k === 'з') { open ? (choice = 'parts', paintChoice())
                                          : commit('parts', false); }
  else if (k === 'i' || k === 'ш') { open ? (choice = 'intact', paintChoice())
                                          : commit('intact', false); }
  else if (k === 'u' || k === 'г') { open ? (choice = 'unclear', paintChoice())
                                          : commit('unclear', false); }
  else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); }
  else if (e.key === 'ArrowRight') { i = Math.min(i + 1, view.length - 1); show(); }
});

const RU = { damaged: 'повреждение', wreck: 'авария', parts: 'разобрана',
             intact: 'целая', unclear: 'не понять' };

function setMode(label) {
  const back = document.getElementById('backpill');
  document.querySelectorAll('.pill.tap').forEach(
    p => p.classList.toggle('on', p.dataset.lab === label));
  if (!label) {
    mode = 'queue'; view = QUEUE; back.classList.add('hid');
  } else {
    const rows = DONE.filter(r => r.label === label);
    if (!rows.length) {
      hint.innerHTML = '<span class="warn">таких меток пока нет</span>';
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


def page(queue_rows: list[dict], counts: dict,
         done_rows: list[dict] | None = None) -> str:
    """HTML страницы. Очередь уезжает в JavaScript одним куском.

    json.dumps со всеми полями сразу, а не подстановка по одному: очередь
    может быть на четыреста кадров, и генерировать четыреста блоков разметки
    ради того, чтобы браузер показывал по одному, бессмысленно.
    """
    out = TEMPLATE.replace("__QUEUE__", json.dumps(queue_rows, ensure_ascii=False))
    out = out.replace("__DONE__",
                      json.dumps(done_rows or [], ensure_ascii=False))
    # Перечень берётся из LABELS, а не списком здесь: захардкоженный кортеж
    # уже разошёлся с метками — «parts» добавили, сюда вписать забыли, и
    # счётчик разобранных на свежей странице показывал ноль.
    for key in LABELS:
        out = out.replace(f'id="c-{key}">0<',
                          f'id="c-{key}">{html.escape(str(counts.get(key, 0)))}<')
    return out
