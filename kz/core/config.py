# -*- coding: utf-8 -*-
"""Central environment configuration.

Each pipeline job runs in its own subprocess and reloads ``.env``. Missing
database settings produce ``DATABASE_URL = None`` rather than an import-time
failure because the public estimator can serve model artifacts without
PostgreSQL. Database-dependent code raises a clear error on first access.
"""

import os

from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

DATABASE_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    if POSTGRES_USER and POSTGRES_PASSWORD and POSTGRES_DB
    else None
)
