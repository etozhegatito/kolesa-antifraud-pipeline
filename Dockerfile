# Контейнер веб-сервиса оценки цены (kz/web).
#
# Что внутри и чего намеренно нет:
#   есть  — код kz/, обученный артефакт модели, зависимости из
#           requirements-web.txt;
#   нет   — собранных объявлений. Данные kolesa.kz не наши, чтобы
#           перевыкладывать их наружу. Модель — производная (веса дерева),
#           по ней объявление не восстановить, и её везём.
#
# Сборка:  docker build -t kz-auto-market-intelligence .
# Запуск:  docker run -p 8000:8000 -e KZ_PUBLIC_DEMO=1 kz-auto-market-intelligence

FROM python:3.13-slim

# PYTHONUNBUFFERED — чтобы логи шли в docker logs сразу, а не когда
# наполнится буфер: без него падение при старте выглядит как молчание.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Зависимости отдельным слоем ДО кода: правка кода не должна заставлять
# заново скачивать catboost и pandas. Docker кэширует слои по порядку,
# и пересобирается всё, начиная с первого изменившегося.
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

COPY kz/ ./kz/
# Production-артефакты в .gitignore и попадают в образ только из локальной
# сборки. CI вместо них создаёт явно непродуктовый smoke-артефакт: он проверяет
# контракт запуска, но не публикует настоящую модель и её fingerprint.
# MODEL_DIR переопределяется только в CI: там создаётся маленький
# синтетический артефакт, чтобы проверить сборку и /api/health, не публикуя
# настоящую модель. Обычная локальная сборка по-прежнему берёт data/models.
ARG MODEL_DIR=data/models
COPY ${MODEL_DIR}/price_model.cbm \
     ${MODEL_DIR}/price_cheap_specialist.cbm \
     ${MODEL_DIR}/price_model.metadata.json ./data/models/

# Не root: если в сервисе найдут дыру, чужой код не должен получить
# полноправного пользователя внутри контейнера.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# Проверка живости для оркестратора: /api/health грузит артефакт модели,
# то есть отвечает «ок» только когда сервис реально умеет считать, а не
# просто когда процесс запустился.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4).status==200 else 1)"

# sh -c нужен, чтобы $PORT подставился: хостинги (fly.io, Render) задают порт
# переменной окружения, а не фиксируют 8000.
CMD ["sh", "-c", "uvicorn kz.web.app:app --host 0.0.0.0 --port ${PORT}"]
