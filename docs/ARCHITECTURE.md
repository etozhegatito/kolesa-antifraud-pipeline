# Architecture

## Design principle

The system separates evidence from conclusions:

> Raw data is append-only; derived conclusions are rebuilt.

Changing a cleaning rule, duplicate detector, or anomaly threshold must never
rewrite what was originally observed. This makes the pipeline auditable and
allows corrected logic to be applied consistently to historical records.

## End-to-end flow

```text
kolesa.kz listing pages
        │
        │ kz/collect/parser.py
        ▼
raw_ads ───────── sightings ───────── photos
   │                    │                 │
   │ enrich             │ status          │ download + pHash
   ▼                    ▼                 ▼
enriched             ad_status       photo_hashes/files
   └────────────────────┬─────────────────┘
                        │ kz/transform/clean.py
                        ▼
                    clean_data
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
 price models      anomaly review    CV research
        │               │                │
        ▼               ▼                ▼
 estimator        manual journal     photo journal
```

## Collection layer

### `kz/collect/parser.py`

The listing parser stores the vehicle passport, first observed price, daily
sightings, and photo URLs. It does not enrich each detail page during the same
loop. Keeping the top-level parser narrow makes request accounting and failure
recovery easier.

Collection safeguards include:

- a shared rolling 24-hour request budget;
- atomic request reservation across processes;
- randomized polite pacing;
- a circuit breaker on HTTP 429 and repeated failures;
- a schema-health gate that detects an empty result caused by HTML drift;
- structured last-run metadata in `logs/parser_last_run.json`;
- per-run card and page limits for controlled smoke tests.

### `kz/collect/check_status.py`

Status checks append lifecycle observations such as active, archived, or
deleted. They do not overwrite the original advertisement. This history is
required for later survival analysis and for avoiding false assumptions that an
unchecked ad is known to be active.

### `kz/collect/enrich.py`

Enrichment visits selected detail pages to obtain seller comments, options,
site badges, drivetrain, and safe structured evidence. It stores only a Boolean
VIN-backed flag when the site explicitly exposes vehicle-history evidence; the
VIN itself is never collected.

The queue prioritizes suspicious rows, reserves capacity for fresh listings,
then prioritizes inexpensive vehicles. Fresh coverage prevents enrichment
presence from becoming a proxy for the target segment.

### Photos

`photo_fetch.py` downloads controlled batches. `photo_dedup.py` computes
perceptual hashes and groups exact cross-listing image copies. DNS preflight
avoids long retries against retired CDN hosts. Image traffic has its own budget.

## Transformation layer

`kz/transform/clean.py` rebuilds `clean_data` from source tables. It performs:

- numeric and categorical normalization;
- missingness indicators;
- impossible-value checks;
- contextual `price_basis` classification for customs, credit, and down-payment
  terms;
- relist grouping without using price;
- rule-based anomaly signals;
- exculpation for disclosed damage or other valid low-price reasons;
- attachment of the latest manual verdict without mutating the journal.

The clean layer is disposable. Raw, enriched, status, and manual-label layers
are durable.

`price_basis` is deliberately high precision. The classifier associates the
saved price with the nearest labelled amount, handles customs negation and
common spelling variants, and returns `ambiguous` when structured and textual
evidence conflict. Training rejects only known non-comparable bases; uncertainty
is not treated as proof that a row is invalid.

## Modelling layer

`kz/ml/train_price_model.py` trains a general CatBoost regressor and a specialist
for inexpensive vehicles. Validation groups relisted copies together and also
holds out later listings by time. The artifact metadata records feature schema,
target policy, data fingerprint, code hash, Git commit, training rows, and
measured metrics.

At inference:

1. the general model predicts every vehicle;
2. predictions below 5M tenge are routed to a specialist trained on actual
   prices below 8M;
3. the selected log-price prediction is transformed back to tenge;
4. conformal residual calibration supplies a range;
5. SHAP values explain the individual result.

Routing is based on the first model's prediction, which is available at serving
time. Routing on the actual price would be target leakage.

## Review layers

`/label` operates on listings and records fraud/legit/unknown verdicts. Its queue
combines rule positives, residual candidates, and a random control sample.
Controls are required to estimate missed cases and recall.

`/damage` operates on individual frames. It records exact English label keys,
optional notes, one or more relative bounding boxes, selection provenance,
dataset split, annotator, version, and review status.

Both journals are atomically updated and snapshotted before mutation. Tests use
`KZ_LABELS_DIR` to redirect writes to a scratch directory.

## Storage

Important PostgreSQL tables include:

| Table | Role |
|---|---|
| `raw_ads` | One immutable vehicle passport per ad ID |
| `sightings` | One observation per ad and day |
| `photos` | Ordered source photo URLs |
| `enriched` | Detail-page fields and fetch status |
| `ad_status` | Append-only lifecycle checks |
| `photo_hashes` | Perceptual image hashes and fetch outcomes |
| `clean_data` | Rebuilt analytical table |

CSV files are retained where they provide append-only audit logs, portable
manual journals, or offline artifacts. PostgreSQL is the operational store;
CSV is not treated as a second mutable database.

## Repository structure

```text
kz/core/       configuration, database, pacing, freshness
kz/collect/    listing, detail-page, status, and photo collection
kz/transform/  deterministic cleaning and data-quality logic
kz/ml/         price, interval, anomaly, monitoring, and CV research
kz/report/     reports plus human-review queue generation
kz/ops/        orchestrated flows and operational status
kz/web/        FastAPI routes, service logic, and HTML interfaces
airflow/       collection and offline DAGs
tests/         offline regression and safe PostgreSQL integration tests
```

## Airflow

Two DAGs represent two operational concerns:

- collection DAG: one budgeted network chain;
- offline DAG: clean, monitor drift, train, calibrate, evaluate, and report.

The network catch-up remains one coordinated task because independent status,
enrichment, and photo tasks could reserve overlapping quotas. The offline graph
places drift monitoring before retraining so the previous artifact is compared
with genuinely new data rather than with itself.

Regression tests verify that DAG dependencies contain every corresponding
pipeline step and preserve the intended order.

## Deployment boundary

The public Docker image contains only inference code and three derivative model
artifacts: the general model, cheap-segment specialist, and metadata. It does
not contain raw listings, the operational database, photos, seller text, or
manual labels. `KZ_PUBLIC_DEMO=1` disables every labelling write endpoint.
