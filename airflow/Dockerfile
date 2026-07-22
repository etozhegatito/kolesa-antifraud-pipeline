# Airflow demo-слой (портфолио/собес) — НЕ путь по умолчанию для
# ежедневного сбора, тот остаётся run_all.py + cron (см. README).
# Этот образ = apache/airflow + зависимости самого пайплайна, чтобы
# BashOperator в DAG мог реально выполнить parser.py/clean.py/... —
# то есть DAG не имитирует пайплайн, а по-настоящему его запускает.
FROM apache/airflow:2.10.5-python3.11

USER root
# Системные библиотеки для headless Chromium (Playwright) на Debian —
# без них parser.py падает при первом же launch() браузера.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libx11-xcb1 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# Airflow 2.10 жёстко требует SQLAlchemy==1.4.54 (свой constraints-файл) —
# конфликтует с нашим "SQLAlchemy>=2.0" (нужен для основного venv/cron,
# трогать requirements.txt из-за одного контейнера не стоит). Внутри ЭТОГО
# образа просто используем версию SQLAlchemy, которую и так тащит Airflow —
# db.py использует только базовые API (create_engine/execute/text),
# совместимые с 1.4.
RUN grep -v '^SQLAlchemy' /tmp/requirements.txt > /tmp/requirements-airflow.txt
# --constraint: официальный способ ставить свои пакеты в образ Airflow,
# не ломая его собственные зафиксированные версии
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
RUN playwright install chromium
