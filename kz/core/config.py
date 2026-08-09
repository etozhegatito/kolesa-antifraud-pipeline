# -*- coding: utf-8 -*-
"""
config.py — единая точка чтения .env. Каждый джоб запускается своим
subprocess'ом (см. run_all.py), общего процесса/памяти между ними нет —
поэтому конфиг читается из файла заново в каждом, а не передаётся
в аргументах.

Почему отсутствие настроек базы больше не падение на импорте. Пайплайну
без базы делать нечего, и раньше строка `os.environ["POSTGRES_USER"]`
честно роняла запуск сразу. Но тот же код импортирует веб-сервис оценки, а
он в облаке работает вообще без базы: вся модель лежит в артефакте, база
нужна только чтобы показать похожие объявления. Падение на импорте не
давало контейнеру даже стартовать.

Теперь отсутствие настроек — это DATABASE_URL = None. Кто без базы жить не
может, получит внятную ошибку при первом обращении (см. kz/core/db.py), а
кто может — просто обойдётся без неё.
"""

import os

from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER     = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_DB       = os.environ.get("POSTGRES_DB")
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.environ.get("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    if POSTGRES_USER and POSTGRES_PASSWORD and POSTGRES_DB
    else None
)
