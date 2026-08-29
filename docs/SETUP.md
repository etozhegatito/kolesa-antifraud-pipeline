# Установка и работа

От пустого каталога до первой оценки цены: что поставить, что запустить, в
каком порядке и что делать, когда сломалось.

Разворачивание сервиса наружу — отдельно, в [DEPLOY.md](DEPLOY.md).

---

## Установка с нуля

Инструкция ниже рассчитана на macOS или Linux. В Windows удобнее использовать
WSL2.

## Шаг 0. Установить внешние программы

Нужны:

- Git;
- Python 3.11 или новее;
- Docker Desktop с запущенным Docker Engine.

Проверка:

```bash
git --version
python --version
docker --version
docker compose version
```

Если какая-либо команда не найдена, сначала установите соответствующую
программу.

## Шаг 1. Скачать проект

```bash
git clone https://github.com/etozhegatito/kolesa-antifraud-pipeline.git
cd kolesa-antifraud-pipeline
```

Проверка:

```bash
pwd
ls
```

В выводе должны быть `README.md`, `requirements.txt`, `docker-compose.yaml` и
Python-файлы проекта.

## Шаг 2. Создать отдельное Python-окружение

```bash
python -m venv .venv
source .venv/bin/activate
```

После активации в начале строки терминала обычно появляется `(.venv)`.

Проверка:

```bash
which python
```

Путь должен вести внутрь папки проекта:

```text
.../kolesa-antifraud-pipeline/.venv/bin/python
```

Перед каждой новой сессией терминала окружение нужно активировать снова:

```bash
source .venv/bin/activate
```

## Шаг 3. Установить Python-зависимости

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Chromium нужен только разрешённому сетевому сборщику:

```bash
playwright install chromium
```

Для тестов и полностью офлайн-анализа браузер не открывается.

## Шаг 4. Создать локальный конфиг

```bash
cp .env.example .env
```

Откройте `.env` и замените `change-me` на собственный локальный пароль:

```dotenv
POSTGRES_USER=admin
POSTGRES_PASSWORD=your-local-password
POSTGRES_DB=market_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Файл `.env` находится в `.gitignore`. Его нельзя публиковать.

## Шаг 5. Запустить PostgreSQL

Убедитесь, что Docker Desktop работает, затем:

```bash
docker compose up -d
```

Проверка:

```bash
docker ps
```

Нужен контейнер:

```text
market_db_container
```

При первом старте на пустом Docker volume PostgreSQL выполняет
`sql/init/01_schema.sql` и создаёт raw-таблицы.

## Шаг 6. Запустить тесты

```bash
python -m pytest tests/ -q
```

Ожидаемый результат:

```text
`95 passed`
```

Эти тесты не доказывают качество модели. Они доказывают, что известные
инженерные ошибки не вернулись:

- отрицания в damage-тексте;
- ложные перезаливы;
- ошибки CSV/PostgreSQL типов;
- неправильная калибровка;
- leakage дублей между фолдами;
- несовпадение train/inference схемы;
- ошибки очереди разметки;
- неправильная матрица ошибок.

---

## Что делать после установки

Выберите свой сценарий.

## Сценарий A. Я проверяю проект как рекрутер или Data Scientist

Без данных можно:

```bash
python -m pytest tests/ -q
```

Затем изучить:

- этот README;
- [MODEL_CARD.md](MODEL_CARD.md);
- `tests/test_pipeline.py`;
- `kz/ml/train_price_model.py`;
- `kz/transform/clean.py`;
- `kz/ops/run_all.py`;
- SQL-схему `sql/init/01_schema.sql`.

Нельзя воспроизвести опубликованные метрики без обучающего среза. Данные
исключены из Git намеренно.

## Сценарий B. У меня есть собственные CSV совместимой схемы

Положите доступные файлы по ожидаемым путям:

```text
data/raw/raw_data.csv
data/raw/sightings.csv
data/raw/photos.csv
data/raw/ad_status.csv
data/enriched/enriched.csv
data/enriched/photo_hashes.csv
```

Необязательные файлы можно пропустить. Затем:

```bash
python -m kz.ops.migrate_to_postgres
python -m kz.transform.clean
python -m kz.report.explore
python -m kz.ml.train_price_model
python -m kz.ml.residual_detector
python -m kz.report.ml_dashboard
python -m kz.report.ml_report
```

Что должно появиться:

```text
PostgreSQL: clean_data
data/clean/clean_data.csv
data/models/price_model.cbm
data/models/price_cheap_specialist.cbm
data/models/price_model.metadata.json
data/models/price_floor.cbm
data/models/price_floor.metadata.json
data/eda/dashboard.png
data/eda/ml_dashboard.png
data/eda/ml_report.html
data/eda/labeling_queue.csv
```

## Сценарий C. PostgreSQL уже заполнен

Полный офлайн-пересчёт:

```bash
python -m kz.ops.run_all --fast
```

Он выполняет:

```text
clean.py → explore.py
```

После этого модели обучаются отдельно:

```bash
python -m kz.ml.train_price_model
python -m kz.ml.residual_detector
python -m kz.report.ml_dashboard
python -m kz.report.ml_report
```

`--fast` не делает сетевых запросов. Он всё равно требует заполненные raw-таблицы
PostgreSQL.

## Сценарий D. У меня есть письменное разрешение на сетевой сбор

Только в этом случае доступны сетевые режимы:

```bash
python -m kz.ops.run_all
```

Порядок:

```text
parser
→ check_status
→ clean, проход 1
→ enrich и photo_dedup
→ clean, проход 2
→ explore
```

Облегчённый режим:

```bash
python -m kz.ops.run_all --light
```

Он собирает листинг и выполняет офлайн-пересборку, но пропускает тяжёлые
per-ad задачи.

`kz/ops/catch_up.py` показывает и дозаполняет пробелы:

```bash
python -m kz.ops.catch_up
python -m kz.ops.catch_up --run
python -m kz.ops.catch_up --run --values
python -m kz.ops.catch_up --run --backfill
python -m kz.ops.catch_up --run --until-done
```

Ограничения частоты и circuit breaker защищают инфраструктуру от случайного
бесконечного цикла. Они не являются разрешением на сбор.

## Сколько запросов делать за сутки

Главная защита от блокировки — не длина пауз, а суточный **объём** запросов с
одного IP. Поэтому у `kz/ops/catch_up.py` есть настраиваемый потолок:

```bash
python -m kz.ops.catch_up --run --backfill --budget 300
```

Порядок приоритета: `--budget N`, затем переменная окружения
`KOLESA_BUDGET`, затем значение по умолчанию 200. При запуске без `--run`
печатается таблица зон и оценка времени.

| Запросов к kolesa за сутки | Зона | На чём основано |
|---|---|---|
| до 100 | спокойно | гонялось многократно, последствий не было |
| до 200 | безопасно (по умолчанию) | рабочая зона, блокировок не наблюдали |
| до 270 | риск | подходит к зафиксированной блокировке |
| больше 270 | высокий риск | выше уже случившейся блокировки |

Границы взяты не из статей, а из собственного опыта проекта: 2026-07-23
домашний IP получил временную блокировку примерно на 270 запросах за сутки.
Это то же правило, что и для порогов детектора — калибровать на своих данных.

Важное ограничение: счётчик видит только `kz/ops/catch_up.py`. Запросы `kz/ops/run_all.py`,
`kz/collect/parser.py` и ручное листание сайта идут с того же IP, но здесь не считаются.
В дни большого дозаполнения не запускайте всё сразу.

## Ритм запросов

Все джобы, обращающиеся к kolesa.kz, используют общий модуль `kz/core/pacing.py`:

- базовая пауза 4–8 секунд между запросами;
- изредка затяжная пауза вместо базовой;
- каждые 15 запросов длинный перерыв на 30–90 секунд.

Пауза никогда не бывает короче нижней границы диапазона, то есть запросов в
час становится **меньше**, чем при равномерных паузах. Это снижение нагрузки
на сайт, а не маскировка: user-agent, отпечаток браузера и IP не подделываются.
Прогон при этом занимает примерно в полтора раза больше времени, и оценка
времени в `kz/ops/catch_up.py` учитывает это честно.

---

## Как получить прогноз для своей машины

Сначала должен существовать обученный артефакт:

```bash
python -m kz.ml.train_price_model
```

Для демонстрации на случайной строке базы:

```bash
python -m kz.ml.predict_price
```

Для своей машины запустите интерактивный Python:

```bash
python
```

Затем:

```python
from kz.ml.train_price_model import load_artifact
from kz.ml.predict_price import estimate

model, metadata = load_artifact()

price = estimate(
    model,
    brand="Toyota",
    model="Camry",
    year=2019,
    engine_volume=2.5,
    mileage_km=90_000,
    engine_type="бензин",
    transmission="автомат",
    body_type="седан",
    condition="б/у",
)

print(f"{price:,.0f} ₸")
```

Чтобы выйти из Python:

```python
exit()
```

Если часть характеристик неизвестна, её можно не указывать. Но чем меньше
информации, тем менее индивидуальным будет прогноз.

---

## Команды и создаваемые файлы

## Четыре команды — это всё, что нужно помнить

```bash
docker compose up -d                        # поднять базу, один раз за сеанс

python -m kz.ops.run_all --collect          # собрать данные (сеть)
python -m kz.report.label_cards --serve     # размечать вердикты
python -m kz.ops.run_all --ml               # пересчитать всё после разметки
python -m kz.web                            # веб-интерфейс
```

Обычный день выглядит так: собрал → разметил → пересчитал → посмотрел
`data/eda/ml_report.html`. Всё остальное ниже — внутренние шаги, которые
оркестратор вызывает сам; по одному их запускают только при отладке.

## Что делает `--collect`

Единственный безопасный путь в сеть. Внутри:

```text
1. parser        свежий листинг: новые объявления и наблюдения цен
2. catch_up      добор пробелов (статусы, обогащение, фото-хэши) ПОРЦИЯМИ
                 под суточным лимитом запросов
3. photo_fetch   фотографии с CDN — другой хост, своя квота
4. photo_features признаки из новых снимков
5. clean → explore → label_cards   офлайн-пересборка
```

Обогащение здесь обязательно: именно оно приносит полный комментарий
продавца, цвет, растаможку и бейдж состояния — то, на чём держится снятие
ложных подозрений.

Полный режим (`run_all` без флагов) теперь является безопасным алиасом
`--collect`: parser и per-ad джобы делят общий суточный лимит. Прежний прямой
путь status/enrich вне счётчика удалён после аудита 29 августа.

## Что делает `--ml`

Одиннадцать шагов, около двух минут, сети не касается.

| Шаг | Что делает | Какой метод |
|---|---|---|
| 1. `clean` | пересобирает `clean_data`, подхватывает новые вердикты | правила + устойчивая статистика |
| 2. `explore` | графики и очередь разметки из трёх слоёв | стратифицированная выборка |
| 3. `label_cards` | карточки под свежий список | — |
| 4. `monitoring` | не разъехались ли данные со старым артефактом | PSI по каждому признаку |
| 5. `train_price_model` | обучает общую модель и специалиста <5M | CatBoost на `log(price)`, grouped CV, out-of-time |
| 6. `residual_detector` | калибрует «ценовой пол» | квантильная регрессия, α=0.10 |
| 7. `price_interval` | строит продуктовый диапазон цены | grouped conformal calibration |
| 8. `ml_dashboard` | графики качества | — |
| 9. `ml_report` | HTML-отчёт | — |
| 10. `evaluate_detector` | precision/recall антифрода | матрица ошибок, правило трёх |
| 11. `survival` | сколько объявление живёт на рынке | Каплан-Мейер и модель Кокса |

Порядок задан зависимостями: мониторинг идёт **до** обучения и сравнивает
данные с предыдущим развёрнутым артефактом. После train такой замер сравнил бы
текущий срез с самим собой. Графики и HTML читают сохранённые артефакты,
поэтому обучение и калибровка идут раньше них, а отчёт требует все модели.

**Шаги 4 и 11 — не украшение.** Мониторинг отвечает, можно ли ещё доверять
модели: рынок меняется, а обученная модель об этом не узнает и будет уверенно
считать по устаревшим закономерностям. Анализ выживаемости отвечает на
продуктовый вопрос «за сколько продастся» и использует метод, который умеет
работать с незавершёнными наблюдениями: большинство объявлений на момент
замера ещё висит, и обычная регрессия по проданным систематически занижала бы
срок.

## Куда смотреть результат

| Файл | Что внутри |
|---|---|
| `data/eda/ml_report.html` | отчёт по модели + подозрительно дешёвые |
| `data/eda/ml_dashboard.png` | качество, важность признаков, остатки |
| `data/eda/label_cards.html` | карточки для разметки |
| `data/models/price_model.metadata.json` | метрики и отпечатки |
| `data/models/price_cheap_specialist.cbm` | модель для базовых прогнозов ниже 5 млн ₸ |

## Редкие команды

Нужны раз в несколько недель или при разборе проблем.

| Команда | Когда |
|---|---|
| `python -m kz.ops.pipeline_status` | посмотреть, что где недозаполнено |
| `python -m kz.ml.learning_curve` | окупается ли дальнейший сбор данных |
| `python -m kz.ml.photo_ablation` | помогают ли фотографии предсказывать цену |
| `python -m kz.ops.db_stats --diff` | сколько строк прибавилось |
| `python -m kz.report.label_cards --dedupe` | свернуть дубликаты в журнале вердиктов |
| `python -m kz.ops.migrate_to_postgres` | вернуть данные в базу из CSV |
| `python -m pytest tests/ -q` | тесты |

## Точечный добор, если нужен именно он

`catch_up` умеет узкие режимы, но в обычной работе его вызывает `--collect`:

```bash
python -m kz.ops.catch_up --run --values      # только обогащение страниц
python -m kz.ops.catch_up --run --backfill    # только средняя цена и бейдж
python -m kz.ops.catch_up --run --budget 300  # свой потолок запросов на сутки
```

---

## Типичные ошибки

## `ModuleNotFoundError`

Причина: виртуальное окружение не активировано.

Решение:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## `KeyError: POSTGRES_USER`

Причина: нет `.env`.

Решение:

```bash
cp .env.example .env
```

Затем заполните значения.

## `connection refused` или `OperationalError` PostgreSQL

Проверьте Docker:

```bash
docker ps
docker compose up -d
```

Проверьте, что `POSTGRES_PORT` в `.env` совпадает с портом в
`docker-compose.yaml`.

## `relation "clean_data" does not exist`

Raw-таблицы ещё не превращены в clean-слой.

Если база заполнена:

```bash
python -m kz.transform.clean
```

Если база пустая, сначала загрузите собственные совместимые данные.

## `Нет обученного артефакта`

Решение:

```bash
python -m kz.ml.train_price_model
```

Для HTML-антифрод панели также нужен:

```bash
python -m kz.ml.residual_detector
```

## `playwright executable doesn't exist`

Только для разрешённого сетевого режима:

```bash
playwright install chromium
```

## Контейнер есть, но схема не обновилась

SQL из `docker-entrypoint-initdb.d` выполняется только при создании пустого
PostgreSQL volume. Изменение SQL-файла не мигрирует уже существующую базу
автоматически. Не удаляйте volume с данными без резервной копии.

## Метрики отличаются от README

Это нормально, если изменились:

- данные;
- дата среза;
- ручные вердикты;
- признаки;
- версия кода.

Истина конкретного запуска находится в:

```text
data/models/price_model.metadata.json
```

## Тесты прошли, но модель плохая

Unit-тесты проверяют код, а не бизнес-качество. Качество проверяется отдельно:

- grouped CV;
- baseline;
- out-of-time holdout;
- сегментные метрики;
- ручная разметка антифрода.

---
