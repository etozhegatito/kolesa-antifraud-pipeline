# -*- coding: utf-8 -*-
"""
config.py — единая точка чтения .env. Каждый джоб запускается своим
subprocess'ом (см. run_all.py), общего процесса/памяти между ними нет —
поэтому конфиг читается из файла заново в каждом, а не передаётся
в аргументах.
"""

import os

from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER     = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB       = os.environ["POSTGRES_DB"]
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.environ.get("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
