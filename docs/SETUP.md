# Installation and operation

## Requirements

- Git
- Python 3.13.x
- Docker Desktop with Compose
- approximately 4 GB of free disk space for the standard environment
- more space only if the optional photo/CV dependencies are installed

The dependency pins are tied to Python 3.13. Do not substitute another Python
version and assume metrics remain reproducible.

## Install from scratch

```bash
git clone https://github.com/etozhegatito/kz-auto-market-intelligence.git
cd kz-auto-market-intelligence

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
docker compose up -d
```

Edit `.env` before using the database outside a local development machine.
Never commit it. PostgreSQL applies `sql/init/01_schema.sql` automatically on a
new Docker volume.

Verify the installation:

```bash
source .venv/bin/activate
python -m pytest tests/ -q
ruff check kz/ tests/ airflow/
python -m kz.ops.pipeline_status
```

Database integration tests refuse to run against a database whose name does not
contain `test`. This prevents a test fixture from truncating production data.

## Quick evaluation for a reviewer

The fastest route requires no local installation: open the
[live estimator](https://kz-auto-market-intelligence.onrender.com/estimate).
The free service may take about a minute to wake after inactivity.

For a local code review:

```bash
source .venv/bin/activate
python -m pytest tests/test_pipeline.py -q
ruff check kz/ tests/ airflow/
python -m kz.web
```

Open `http://127.0.0.1:8000`. A public clone contains trained demo artifacts but
not raw marketplace data or the private operational database.

## Main commands

Only four commands are needed for normal work:

```bash
python -m kz.web
python -m kz.ops.run_all --collect
python -m kz.ops.run_all --ml
python -m kz.ops.pipeline_status
```

### `python -m kz.web`

Starts the single local application:

- `/estimate`: price estimate and explanation;
- `/label`: market-anomaly verdicts;
- `/damage`: photo labels and bounding boxes;
- `/api/docs`: OpenAPI documentation;
- `/api/health`: loaded-model metadata.

Manual UI tests must use a scratch journal:

```bash
KZ_LABELS_DIR=/tmp/kz-label-test python -m kz.web
```

Never delete a whole file under `data/` to clean up a test.

### `python -m kz.ops.run_all --collect`

Runs the only coordinated network path. It uses a shared rolling 24-hour
request budget across parser, status, enrichment, and related jobs. The parser
stops on HTTP 429, repeated failures, exhausted budget, or schema-health errors.

Network collection is not required to review or run the public estimator. Use
it only when access is permitted. Do not use VPNs, proxies, mobile tethering,
or IP rotation to evade restrictions.

Useful controlled variants:

```bash
KOLESA_MAX_CARDS=10 python -m kz.ops.run_all --light
KOLESA_START_PAGE=1 KOLESA_MAX_PAGES=3 python -m kz.ops.run_all --light
python -m kz.ops.catch_up --run --values --budget 20
python -m kz.ops.catch_up --run --values --until-done --budget 100
```

`--light` runs only the budgeted listing parser. `--collect` coordinates the
complete collection chain. Request budgets are rolling windows, not values that
reset at midnight.

### `python -m kz.ops.run_all --ml`

Runs the deterministic offline chain:

1. rebuild the clean layer;
2. monitor drift against the previous training reference;
3. train general and specialist price models;
4. save grouped and temporal OOF predictions;
5. compute grouped-bootstrap stability and segment metrics;
6. calibrate price intervals;
7. build residual anomaly candidates;
8. generate reports and survival diagnostics.

Do not interrupt artifact publication halfway through. Compare new metadata
with `docs/MODEL_CARD.md` before accepting a dependency or feature change.

### `python -m kz.ops.pipeline_status`

Shows data freshness, useful enrichment coverage, queue backlogs, request
budget usage, latest parser status, model age, and missing artifacts. Run it
before deciding whether the next step is collection, enrichment, labelling, or
offline retraining.

## Optional photo research environment

Photo experiments require larger dependencies:

```bash
pip install -r requirements-photos.txt
python -m kz.ml.photo_clip --validate
python -m kz.ml.photo_damage
python -m kz.ml.photo_dataset
```

The CV workflow should ideally use a separate environment from the pinned
tabular production stack. Current supervised results are quarantined until
legacy labels are visually reviewed.

## Run an estimate from Python

```python
from kz.web.service import full_estimate

result = full_estimate(
    {
        "brand": "Toyota",
        "model": "Camry",
        "age": 8,
        "mileage_km": 95_000,
        "engine_volume": 2.5,
        "engine_type": "бензин",
        "transmission": "автомат",
        "body_type": "седан",
        "condition": "б/у",
        "photos_count": 8,
    },
    asking_price=11_000_000,
    text="One owner, regularly maintained, service records available.",
)
print(result["fair_price"], result["range_low"], result["range_high"])
```

The category values above intentionally match the source-market vocabulary used
by the trained artifact. The web interface displays English labels and submits
these internal values automatically.

## Public Docker image

```bash
docker build -t kz-auto-market-intelligence .
docker run --rm -p 8000:8000 \
  -e KZ_PUBLIC_DEMO=1 \
  kz-auto-market-intelligence
curl -s http://127.0.0.1:8000/api/health
```

Public mode has no PostgreSQL dependency and disables all human-label mutation
routes. See [DEPLOY.md](DEPLOY.md) for the complete boundary.

## Generated outputs

| Path | Purpose |
|---|---|
| `data/models/price_model.cbm` | General price model |
| `data/models/price_cheap_specialist.cbm` | Cheap-segment specialist |
| `data/models/price_model.metadata.json` | Training contract and metrics |
| `data/eda/price_model_oof.csv` | Saved honest OOF predictions |
| `data/eda/mape_stability.json` | Bootstrap and segment stability |
| `data/eda/ml_report.html` | Human-readable ML report |
| `data/manual_labels.csv` | Durable anomaly verdict journal |
| `data/photo_labels.csv` | Durable frame and bounding-box journal |
| `logs/parser_last_run.json` | Structured collection status |

## Troubleshooting

### `ModuleNotFoundError`

Activate the project environment:

```bash
source .venv/bin/activate
which python
python --version
```

The interpreter should point inside `.venv` and report Python 3.13.

### Missing PostgreSQL environment variables

Copy `.env.example` to `.env` and fill the values. Model-only public inference
does not require a database, but collection, cleaning, and comparable listings do.

### Connection refused / `OperationalError`

```bash
docker compose up -d
docker compose ps
docker compose logs postgres
```

Wait for the health check before retrying.

### `relation "clean_data" does not exist`

The schema may exist without derived data. Load compatible source data, then run:

```bash
python -m kz.transform.clean
```

### No trained artifact

Run the full offline chain after `clean_data` is populated:

```bash
python -m kz.ops.run_all --ml
```

### Playwright browser executable missing

```bash
python -m playwright install chromium
```

### A container exists but schema changes are absent

Docker initialization scripts run only when a volume is first created. Apply a
targeted migration or create a new disposable development volume; never destroy
an operational data volume casually.

### Metrics differ from the README

Check `data/models/price_model.metadata.json`, the training timestamp, Git dirty
flag, data fingerprint, and training row count. README metrics describe the
published artifact, not every local experimental snapshot.

### Tests pass but the model is weak

Tests verify software invariants, leakage controls, and failure handling. They
cannot manufacture missing condition signal or guarantee a lower MAPE. Model
quality must be measured separately on saved OOF and temporal predictions.
