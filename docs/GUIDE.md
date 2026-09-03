# Guide: understanding and operating the project

## Start here

For a quick review:

```bash
source .venv/bin/activate
docker compose up -d
python -m kz.web
```

Open:

- `http://127.0.0.1:8000/estimate` for a vehicle estimate;
- `http://127.0.0.1:8000/label` for anomaly verdicts;
- `http://127.0.0.1:8000/damage` for photo labels.

After manual labels change, run:

```bash
python -m kz.ops.run_all --ml
```

Then inspect `data/eda/ml_report.html` and the metadata JSON. Labelling pages
are local-only because they write manual ground truth.

## What the project does

The system turns noisy used-car advertisements into an auditable price model:

```text
collect evidence → enrich selected rows → rebuild clean data
→ group relists → validate honestly → train → calibrate
→ explain predictions → review anomalies
```

The estimator answers “What is a plausible first advertised price for this
vehicle in Almaty?” It does not know the final negotiated sale price.

## Data flow in plain language

1. The parser reads listing cards and stores their original facts.
2. A sighting table records that the same ad was observed on another day.
3. Controlled jobs inspect detail pages, status, and photos.
4. Cleaning rebuilds one analytical table from those durable layers.
5. Relisted copies of one physical vehicle receive one group ID.
6. Models train on log price with group-safe folds.
7. Every training row gets an honest out-of-fold prediction.
8. Those residuals calibrate intervals and anomaly thresholds.
9. Humans review uncertain listing and photo cases.

Raw evidence is not overwritten when logic changes. Only derived tables and
reports are rebuilt.

## The essential mathematics

### Log-price target

Instead of predicting price `p` directly, the model predicts:

```text
z = ln(p)
```

The final output is:

```text
p_hat = exp(z_hat)
```

Why? A 20% pricing error has a comparable meaning for a cheap and expensive
vehicle in log space, and multiplicative effects become approximately additive.

### MAPE

For one row:

```text
APE = |actual - predicted| / actual
```

For `n` rows:

```text
MAPE = (APE_1 + ... + APE_n) / n × 100%
```

Example: actual price is 10M tenge and prediction is 8M. The absolute error is
2M and APE is `2 / 10 = 0.20 = 20%`.

MAPE is easy to explain but strongly penalizes errors on inexpensive cars. The
project therefore reports price segments and median APE as well.

### Median APE

Sort all row-level APE values and select the middle one. This describes the
typical listing. The current median is 13.81%, much lower than 21.36% MAPE,
which tells us that a smaller tail of difficult rows raises the mean.

### MAE

```text
MAE = mean(|actual - predicted|)
```

MAE is measured in tenge and emphasizes expensive vehicles. It answers a
different business question from MAPE.

### R-squared

```text
R² = 1 - model_squared_error / mean_baseline_squared_error
```

The project calculates it on log price. The current value, about 0.935, means
the model explains most log-price variation, but R-squared alone does not tell a
seller the typical percentage error.

### Baseline

A baseline is the simplest credible alternative. Here it predicts a robust
make/model/year group price. If CatBoost cannot beat it on the same split, the
extra complexity is not justified.

### Grouped cross-validation

Suppose one Camry is removed and reposted under a new ad ID. Random splitting
might train on the first copy and validate on the second. The metric would look
better because the model has effectively seen the answer.

Grouped CV treats every relist group as indivisible:

```text
vehicle group A → train or validation, never both
```

OOF means each row was predicted by a fold that did not train on its group.

### Out-of-time evaluation

The last period is held out. Training uses earlier listings and evaluation uses
later ones. This is closer to deployment and exposes market drift.

### Bootstrap uncertainty

One MAPE value is an estimate from one dataset. Grouped bootstrap repeatedly
resamples vehicle groups and recomputes MAPE. The distribution gives a standard
deviation and confidence interval.

The current 95% interval is 20.87%–21.87%. A change from 21.40% to 21.44% is
well inside ordinary variation and should not be called degradation.

## Model logic

### Feature contract

The model uses fields available both in training and at prediction time. The
marketplace's own average price is excluded because it derives from the target.
Partially enriched text/options are excluded until the estimator accepts and
validates the same inputs.

### General model and cheap specialist

The general model predicts first. A prediction below 5M tenge is routed to a
specialist trained on actual prices below 8M. The wider training band smooths
the boundary. Actual price is never used to choose the route.

### Price interval

The interval is calibrated from honest OOF residuals. Roughly 80% empirical
coverage means that about eight out of ten held-out listing prices fell inside
their intervals. It is not a promise about a negotiated transaction.

### Explanation

SHAP decomposes one log-price prediction into a base value plus feature
contributions:

```text
prediction = base + contribution_1 + ... + contribution_k
```

The web service converts log contributions into approximate percentage changes
so the result is readable.

## Anomaly layers

### Deterministic rules

Rules catch known suspicious patterns such as unexplained extreme cheapness,
inconsistent relists, or exact photos reused across different vehicles. Rules
are explainable but incomplete.

### Residual detector

A quantile model estimates a plausible lower price boundary. A listing below it
becomes a candidate unless detail-page evidence explains the price.

### Unsupervised diagnostics

Isolation Forest and similar tools can rank unusual feature combinations, but
“unusual” is not “fraud.” They are diagnostic layers rather than a final judge.

### Human ground truth

The `/label` queue includes detector positives, residual candidates, and random
controls. Controls are required to estimate misses. The final fraud/legit/unknown
verdict lives in an append-preserving CSV journal.

## Computer vision logic

The current task is not “Does the car look old?” It is “Is there localized
impact or deformation?” The protocol separates:

- `damaged`: a local impact/dent that can be boxed;
- `wreck`: a destroyed assembly where a local box is meaningless;
- `parts`: a dismantled vehicle or removed component;
- `intact`: no impact/dent, even if rust or scuffs are present;
- `unclear`: insufficient visual evidence.

Rust belongs to `intact` only for this impact task and should be recorded in the
comment. Earlier wording mixed these meanings, so 47 legacy `damaged` frames are
quarantined until manual review.

CV must be evaluated by independent listings and exact-photo components, not by
frames. ROC-AUC and PR-AUC are both required because positive damage examples
are rare. Active-learning rows cannot double as the final random test set.

## Why 18% MAPE is difficult

Vehicles at 5M tenge and above already sit near 16% MAPE. Vehicles below 5M are
near 29% and create most total percentage error. The 21+-year, below-5M segment
is the sharpest concentration.

Repeated collection of the same listing-card fields reached a plateau. The
remaining signal is likely condition evidence from detail-page descriptions,
options, and photos. To reach roughly 18% overall while the stronger segment
stays constant, the cheap segment must move toward about 20.5%.

## What has already been rejected

Do not repeat these ideas without new evidence or a different experimental
design:

- extra listing-table feature groups;
- title/description text on sparse or inconsistent coverage;
- ResNet50 embeddings for price;
- full-frame CLIP as a damage shortcut;
- routing by actual price;
- global prediction multiplier that improves MAPE through systematic bias;
- wider pHash thresholds that merge common studio photos;
- mixing other cities without a validated geographic product contract.

See [FINDINGS.md](FINDINGS.md) for measurements and failure reasons.

## Command reference

```bash
source .venv/bin/activate
docker compose up -d

python -m kz.ops.pipeline_status
python -m kz.ops.run_all --collect
python -m kz.ops.run_all --ml
python -m kz.web

python -m pytest tests/ -q
ruff check kz/ tests/ airflow/
```

Optional research:

```bash
python -m kz.ml.mape_stability
python -m kz.ml.photo_clip --validate
python -m kz.ml.photo_damage
python -m kz.ml.photo_dataset
python -m kz.report.photo_labels --stats
```

## Current state at a glance

- 12,639 collected listings;
- 12,455 model rows;
- 21.36% grouped routed MAPE;
- 13.81% median APE;
- 23.24% out-of-time MAPE;
- approximately 80% interval coverage;
- 179 anomaly candidates and 111 verdicts, with no confirmed fraud;
- 47 legacy damaged frames awaiting definition review;
- read-only public demo deployed on Render Free.
