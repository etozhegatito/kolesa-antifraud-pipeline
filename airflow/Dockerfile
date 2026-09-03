# Airflow portfolio layer, not the default daily execution path. The default
# remains run_all.py plus cron. This image adds project dependencies so each
# BashOperator executes the real pipeline rather than simulating it.
FROM apache/airflow:2.10.5-python3.11

USER root
# Debian system libraries required by Playwright's headless Chromium.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libx11-xcb1 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# Airflow 2.10 pins SQLAlchemy 1.4.54 through its constraints. The main runtime
# uses SQLAlchemy 2.x, while this isolated image keeps Airflow's compatible
# version; db.py only relies on APIs shared by both versions.
RUN grep -v '^SQLAlchemy' /tmp/requirements.txt > /tmp/requirements-airflow.txt
# --constraint is the supported way to add packages without breaking Airflow's pins.
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
RUN playwright install chromium
