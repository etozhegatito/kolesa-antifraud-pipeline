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

Запуск:  python -m kz.report.label_cards            → data/eda/label_cards.html
         python -m kz.report.label_cards --serve    → то же + локальный сервер, который
                                            ДОПИСЫВАЕТ вердикты в журнал сразу
                                            при нажатии (рекомендуемый режим)
         python -m kz.report.label_cards --all      → включить и residual-кандидатов
                                            из labeling_queue.csv, не только
                                            правиловых подозрительных

КАК СОХРАНЯЮТСЯ ВЕРДИКТЫ (три уровня, каждый со своей задачей):
  1) localStorage браузера — мгновенно, переживает перезагрузку и закрытие
     вкладки. Работает всегда, даже при открытии файла напрямую.
  2) data/manual_labels.csv — источник истины, читается clean.py. Пишется
     ТОЛЬКО в режиме --serve: страница, открытая как file://, писать на

СТРУКТУРА ПАКЕТА. Файл дорос до 1239 строк и делал четыре разные вещи
сразу: читал базу, генерировал HTML, вёл журнал вердиктов и поднимал
HTTP-сервер. Разнесено по ответственностям, чтобы каждую можно было читать
и проверять отдельно:

    queue.py     что показывать — выборка из базы
    render.py    как показывать — HTML, без базы и без записи на диск
    journal.py   куда писать вердикты — единственная точка правки разметки
    server.py    локальный сервер, чтобы страница могла сохранять выбор

Имена ниже переэкспортированы, поэтому `from kz.report import label_cards`
продолжает работать как раньше.
"""

from kz.report.label_cards.journal import (          # noqa: F401
    BASE_HEADER,
    LABELS_CSV,
    LABELS_PREV,
    STRATUM_COLS,
    VERDICTS,
    dedupe_journal,
    journal_facts,
    journal_header,
    read_journal,
    upsert_verdict,
    write_journal,
)
from kz.report.label_cards.queue import QUEUE_CSV, load_rows      # noqa: F401
from kz.report.label_cards.render import (            # noqa: F401
    DEAD_HOSTS,
    FLAG_HELP,
    OUT_HTML,
    PRICE_BANDS,
    build,
    card_html,
    fmt,
    money,
    photo_src,
    price_band,
    price_verdict_hint,
)
from kz.report.label_cards.server import serve        # noqa: F401
