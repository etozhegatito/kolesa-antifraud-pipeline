# KZ Auto Market Intelligence

**English** | [Русский](README_RU.md)

An end-to-end data and machine-learning system for the Almaty used-car
market: collection, data quality, duplicate-safe validation, price estimation,
market-anomaly review, and experimental visual-condition analysis.

> Enter a car's make, model, year, mileage, engine, transmission, and body
> type. The service returns a fair listing-price estimate, a calibrated range,
> the main factors behind the estimate, and input-quality warnings.

[**Open the live demo**](https://kz-auto-market-intelligence.onrender.com/estimate)
· [Model card](docs/MODEL_CARD.md)
· [Verified findings](docs/FINDINGS.md)

## The project in 30 seconds

| Question | Answer |
|---|---|
| What problem does it solve? | Estimates a fair *listing* price for a used car in Almaty and sends unusually low listings to human review. |
| What is deployed? | A read-only estimator backed by two trained CatBoost models. It does not need the source database for inference. |
| How much data? | 12,639 collected listings; 12,455 rows used by the current model. |
| Main result | **21.36% grouped out-of-fold MAPE**, 13.81% median APE, 23.24% out-of-time MAPE. |
| Where is the remaining error? | Cars below ₸5M: 29.01% MAPE and 55.5% of total percentage error. Cars at ₸5M+: 16.07% MAPE. |
| Is computer vision in production? | No. Earlier supervised CV results were withdrawn after label-definition drift was found. The live price estimate does not claim to inspect photos. |
| Engineering quality | 255 offline tests plus 6 PostgreSQL integration tests, Ruff, Docker health smoke, and GitHub Actions. |

The free demo can sleep after inactivity. Its first request may therefore take
about a minute; later requests are fast.

## Why this project exists

A marketplace price is not determined only by make, model, and year. Mileage,
engine, transmission, body type, vehicle condition, seller description, and
photos can all matter. Marketplace data is also noisy: the same vehicle may be
relisted, prices can change, and exceptionally cheap ads may be damaged cars,
incomplete listings, bait prices, or genuine bargains.

This project turns that source into an auditable pipeline:

1. collect listing snapshots without overwriting the original evidence;
2. enrich a controlled subset with fields from the full listing page;
3. rebuild a clean analytical table from immutable raw data;
4. group relisted vehicles so copies cannot leak across train and validation;
5. train and compare a general model and a cheap-car specialist;
6. calibrate price intervals and low-price review thresholds on out-of-fold residuals;
7. keep uncertain anomaly and photo decisions in human-review queues;
8. publish only the read-only estimator, never collection or labeling tools.

The system estimates the **first observed listing price**, not the final sale
price. That distinction is part of the model contract and is stored in its
metadata.

## Try the demo

Open the [live estimator](https://kz-auto-market-intelligence.onrender.com/estimate)
and choose **Price estimate**. A realistic example is:

```text
Make: Toyota
Model: Camry
Year: 2019
Mileage: 90,000 km
Engine: 2.5 L petrol
Transmission: automatic
Body: sedan
Condition: used
```

The response contains:

- a point estimate in Kazakhstani tenge;
- a calibrated lower and upper range;
- the model route that answered;
- plain-language factors that moved the estimate;
- warnings for missing or unusual input values.

The public mode intentionally disables `/label` and `/damage`. Those pages
write human ground truth and must not be exposed to anonymous users. With no
PostgreSQL database, comparable-listing decoration is omitted, but the model
estimate still works.

Check the deployed artifact directly:

```bash
curl -s https://kz-auto-market-intelligence.onrender.com/api/health
```

The response includes the training timestamp, number of training rows, current
validation MAPE, and whether public-demo safeguards are active.

## Current measured results

These numbers come from the model trained on 3 September 2026. They are based
on saved out-of-fold predictions, not predictions on training rows.

| Validation view | MAPE | Median APE | Notes |
|---|---:|---:|---|
| Grouped OOF, routed model | **21.36%** | **13.81%** | Primary model-selection estimate |
| Grouped OOF, general model only | 21.38% | 13.94% | Routing changes little on this snapshot |
| Grouped OOF, simple baseline | 30.70% | 14.21% | Median by make + model + year |
| Out-of-time, routed model | **23.24%** | 14.82% | Later listings held out by time |
| Out-of-time, baseline | 35.01% | 15.25% | Same temporal holdout |

The routed model's grouped MAPE has a 95% grouped-bootstrap interval of
**20.87%–21.87%**. Its improvement over the general model is only -0.03
percentage points, and the paired confidence interval crosses zero. The
specialist remains a researched production route, but the current snapshot
does not justify claiming a meaningful overall win.

### Error by price

| Actual listing price | Rows | MAPE | Share of total percentage error |
|---|---:|---:|---:|
| Below ₸5M | 5,089 | **29.01%** | **55.5%** |
| ₸5M and above | 7,366 | **16.07%** | 44.5% |

The 18% overall MAPE target is a research gate, not a promise. If the stronger
segment remains unchanged, the below-₸5M segment must improve to roughly
20.5%. Repeated experiments show that collecting more listing cards alone is
not enough; the missing signal is likely the physical condition of older,
inexpensive cars.

### Error by age

| Vehicle age | Rows | MAPE |
|---|---:|---:|
| 0–5 years | 3,350 | 16.72% |
| 6–10 years | 1,581 | 15.74% |
| 11–20 years | 3,353 | 18.37% |
| 21+ years | 4,171 | **29.62%** |

The sharpest intersection is **21+ years and below ₸5M**: 3,518 rows, 31.05%
MAPE, and 41.1% of all percentage error. The roadmap therefore prioritizes
condition evidence instead of treating every car older than five years as a
single difficult class.

## How the system works

```text
kolesa.kz listing pages
        │ parser: vehicle fields, observations, photo URLs
        ▼
raw_ads + sightings                 immutable evidence
        │
        ├── enrich: description, condition badge, options, VIN-backed flag
        ├── check_status: active / archived / deleted
        ├── photo_dedup: perceptual hashes
        └── photo_fetch: controlled image downloads
        ▼
enriched + status + photo metadata
        │ clean: deterministic rebuild
        ▼
clean_data
        │
        ├── duplicate/relist grouping
        ├── grouped and temporal validation
        ├── general CatBoost model
        ├── specialist for predicted prices below ₸5M
        └── interval and anomaly calibration
        ▼
price artifacts + reports + review queues
```

The governing rule is: **raw evidence is append-only; conclusions are
recomputed**. If a cleaning or anomaly rule changes, the complete clean layer
is rebuilt from the original observations. This makes corrections
reproducible and prevents past conclusions from mutating their source.

## Modelling logic

### Target and features

The model learns `log(first observed listing price in KZT)`. The logarithm
reduces the dominance of very expensive vehicles and turns many multiplicative
price relationships into easier additive relationships. Predictions are
transformed back to tenge for the user.

The deployed models use 13 features:

- numeric: vehicle age, mileage, engine volume, and photo count;
- flags: mileage missing, VIP listing, and monthly-price display;
- categorical: make, model, engine type, transmission, body type, condition.

Marketplace-provided average price and listing category are excluded because
they leak target-derived information. Description, options, and image features
are excluded until their coverage and train/serve consistency are sufficient.

### Duplicate-safe validation

A vehicle can be deleted and relisted under a new ad ID. A random split could
put one copy in training and another in validation, letting the model remember
the answer. The project detects relist groups without using price in the group
key and keeps every complete group in one fold.

The primary metric is grouped out-of-fold MAPE:

```text
APE_i = |actual_i - prediction_i| / actual_i
MAPE  = mean(APE_i) × 100%
```

Median APE is reported because MAPE is pulled upward by a small tail of large
relative errors. Out-of-time validation then asks a separate question: how the
model behaves on listings that appeared later.

### Routed inference and calibrated ranges

The general model predicts every row first. If its prediction is below ₸5M,
the request is routed to a specialist trained on the wider actual-price band
below ₸8M. Routing uses only information available at inference time; using
the unknown actual price would be target leakage.

The price range and suspiciously-low threshold are calibrated on held-out OOF
residuals. The anomaly layer creates a review queue; it does not declare that a
seller is fraudulent. A visibly damaged car at an honest low price is a
legitimate listing, not fraud.

## Computer vision status

Computer vision is **experimental and not deployed**. The repository contains
image download, perceptual-duplicate grouping, CLIP features, tiled crops,
bounding-box labeling, COCO export, grouped CV, and ROC-AUC/PR-AUC evaluation.
However, a label-definition audit found that the historical `damaged` class
mixed impact/dent damage with rust, scratches, dirt, and paint defects.

All 47 legacy `damaged` frames were placed in `needs_review` without deleting
or automatically relabeling user work. Earlier supervised CV metrics were
withdrawn, and pending rows are excluded from training. This does not affect
the price model because CV never passed its gate and was never connected to
price inference.

The next valid CV milestone is:

1. finish visual review under one written label policy;
2. collect 150–200 independent positive ads, not merely frames;
3. preserve a random audit split before active-learning ranking;
4. group by ad and exact-photo duplicate components;
5. beat the age+price baseline with a positive lower bound for paired bootstrap delta AUC;
6. only then test whether a condition score improves price MAPE.

## Repository map

```text
kz/core/       configuration, database, pacing, freshness
kz/collect/    collection, enrichment, status, photo download/dedup
kz/transform/  clean-table rebuild, damage text parsing, data quality
kz/ml/         price, intervals, anomaly residuals, monitoring, CV research
kz/report/     reports and human-review queues
kz/ops/        full pipeline, catch-up scheduler, operational status
kz/web/        FastAPI application and read-only public estimator
airflow/       collection and offline DAGs
tests/         offline regression and PostgreSQL integration suites
deploy/models/ production demo artifacts only; no source marketplace rows
docs/          architecture, findings, setup, model card, and deployment
```

## Run locally

```bash
git clone https://github.com/etozhegatito/kz-auto-market-intelligence.git
cd kz-auto-market-intelligence
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # choose your own PostgreSQL password
docker compose up -d          # applies the schema automatically
python -m pytest tests/ -q
ruff check kz/ tests/ airflow/
python -m kz.web
```

Open `http://127.0.0.1:8000`. The local app contains the estimator plus the
protected labeling workflows. The public deployment exposes only estimation.

The usual workflow is:

```bash
python -m kz.ops.run_all --collect   # the only coordinated network path
python -m kz.ops.run_all --ml        # deterministic offline rebuild and train
python -m kz.ops.pipeline_status     # freshness, backlogs, last-run status
```

Collection uses a shared rolling 24-hour request budget, pacing, a 429 circuit
breaker, atomic reservation, and an HTML-schema health gate. Polite pacing
does not replace permission. Do not use VPNs, proxies, mobile tethering, or IP
rotation to bypass access restrictions.

## Run the public container

Only trained artifacts required for inference are included. Raw listings and
photos are not.

```bash
docker build -t kz-auto-market-intelligence .
docker run --rm -p 8000:8000 \
  -e KZ_PUBLIC_DEMO=1 \
  kz-auto-market-intelligence
curl -s http://127.0.0.1:8000/api/health
```

The image runs as a non-root user and one Uvicorn worker so it remains within
the 512 MB free-instance limit. `render.yaml` defines the free service and its
model-loading health check.

## Reproducibility and safety

- Dependency versions and Python 3.13 are pinned.
- Each model records its data fingerprint, code hash, Git commit, training
  size, metrics, and target policy.
- Raw ads are append-only; derived tables are rebuilt.
- Tests refuse to use a database whose name does not contain `test`.
- CI runs offline tests, PostgreSQL tests, Ruff, coverage, and a real Docker
  model-load health smoke.
- Public mode disables every endpoint that can mutate human labels.
- No phone numbers are collected, and the VIN itself is never stored.
- Raw listings, descriptions, photos, labels, credentials, reports, and the
  private textbook are not published.

The `.cbm` files contain trained tree weights. They are derivative inference
artifacts, not a browsable copy of the source listings.

## Documentation

This README is the complete English entry point. Deeper engineering notes are
currently maintained in Russian:

| Document | Purpose |
|---|---|
| [Overview](docs/OVERVIEW.md) | Scope, current state, constraints, roadmap |
| [Architecture](docs/ARCHITECTURE.md) | Data flow, storage, repository layout, Airflow |
| [Modelling](docs/MODELLING.md) | Features, grouped CV, temporal validation, metrics |
| [Verified findings](docs/FINDINGS.md) | Positive and negative experiments, including withdrawn claims |
| [Model card](docs/MODEL_CARD.md) | Contract and measured behavior of the current artifact |
| [Anomaly review](docs/ANTIFRAUD.md) | Rules, residuals, ground truth, terminology |
| [Setup](docs/SETUP.md) | Installation, commands, troubleshooting |
| [Deployment](docs/DEPLOY.md) | Public-mode boundary and Render deployment |
| [Glossary](docs/GLOSSARY.md) | Project terminology |

## What the project does not claim

- It does not predict the final transaction price.
- It does not diagnose a vehicle's mechanical condition.
- It does not automatically accuse a seller of fraud.
- It does not currently use seller text or uploaded photos in the live price answer.
- It does not guarantee that a vehicle will sell inside the predicted range.
- It does not claim that the 18% MAPE target has been reached.

## Author

Built and maintained by **Sanzhar Bakirbayev** as a production-minded data
science and data-engineering portfolio project.
