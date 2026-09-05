# Price-estimation web service container (kz/web).
#
# Included: kz/ code, trained model artifacts, and requirements-web.txt.
# Excluded: collected listings, seller text, images, and manual labels.
# The model is a derivative tree-weight artifact and cannot be browsed as a
# copy of the source listings.
#
# Build: docker build -t kz-auto-market-intelligence .
# Run:   docker run -p 8000:8000 -e KZ_PUBLIC_DEMO=1 kz-auto-market-intelligence

FROM python:3.13-slim

# Stream logs immediately so startup failures do not look like a silent hang.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies before copying source so source-only changes reuse the
# expensive CatBoost/pandas layer.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY kz/ ./kz/
# The public repository contains only the derivative point and interval
# artifacts required for inference. CI overrides MODEL_DIR with explicitly
# synthetic smoke artifacts; it validates startup and /api/health without
# pretending that the smoke models are production models.
ARG MODEL_DIR=deploy/models
COPY ${MODEL_DIR}/price_model.cbm \
     ${MODEL_DIR}/price_cheap_specialist.cbm \
     ${MODEL_DIR}/price_model.metadata.json \
     ${MODEL_DIR}/price_interval_lower.cbm \
     ${MODEL_DIR}/price_interval_upper.cbm \
     ${MODEL_DIR}/price_interval.metadata.json ./data/models/

# Run without root privileges to reduce the impact of a service compromise.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# The health endpoint loads the model, so healthy means inference is available,
# not merely that the process has started.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4).status==200 else 1)"

# Hosting platforms provide the listening port through $PORT.
CMD ["sh", "-c", "uvicorn kz.web.app:app --host 0.0.0.0 --port ${PORT}"]
