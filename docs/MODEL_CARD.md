# Model card: vehicle listing-price estimator

## Intended use

The model estimates the first observed advertised price of a passenger vehicle
in Almaty, Kazakhstan. It is designed for exploratory seller guidance,
market-monitoring research, and a portfolio demonstration of leakage-aware
machine learning.

It is not a dealer appraisal, loan decision, mechanical inspection, final sale
price, or guarantee that a vehicle will sell inside the displayed range.

## Data

- Source market: kolesa.kz listings scoped to Almaty.
- Collected listings: 12,799.
- Rows used by the current trained artifact: 12,639.
- Target: first observed listing price in KZT.
- Target cleaning: explicit uncleared-cash, credit-price, down-payment, and
  missing-engine-and-gearbox parts prices are excluded; ambiguous rows are retained.
- Training timestamp: 5 September 2026.
- Raw listings, seller descriptions, photos, and manual labels are private and
  are not included in the public repository.

Duplicate and relist groups are derived without using price. Complete groups
remain on one side of validation to prevent the same physical vehicle from
appearing in both training and evaluation.

## Model

The production route contains two CatBoost regressors trained on log price:

1. a general model for every request;
2. a cheap-vehicle specialist trained on actual prices below 8M tenge.

The general model predicts first. If that prediction is below 5M tenge, the
specialist answers. This rule uses only inference-time information; routing on
the unknown actual price would leak the target.

Features:

- numeric: age, mileage, engine displacement, and photo count;
- missing/listing flags: mileage missing, VIP status, monthly-price display;
- categorical: make, model, fuel, transmission, body style, and condition.

Kolesa's own average price and listing category are excluded because they are
target-derived. Text and photo features are excluded from the current price
artifact because coverage and train/serve parity are not yet sufficient.

## Validation results

Measured on the 5 September 2026 artifact:

| Validation view | MAPE | Median APE | R-squared on log price |
|---|---:|---:|---:|
| Grouped OOF, routed | **21.48%** | **13.88%** | **0.935** |
| Grouped OOF, general only | 21.62% | 14.02% | 0.935 |
| Grouped OOF, make/model/year baseline | 30.49% | 14.16% | 0.850 |
| Out-of-time, routed | **22.99%** | 14.41% | 0.929 |
| Out-of-time, baseline | 34.31% | 15.00% | 0.841 |

Grouped bootstrap for routed MAPE:

- standard deviation: about 0.26 percentage points;
- 95% interval: **20.99%–22.02%**.

The routed model changes grouped MAPE versus the general model by -0.13
percentage points. Its paired 95% interval is -0.29 to +0.02 points, and the
out-of-time interval also crosses zero. Routing therefore remains experimental
rather than a statistically supported improvement claim.

## Segment behavior

| Segment | Rows | MAPE |
|---|---:|---:|
| Price below 5M tenge | 5,173 | **29.22%** |
| Price at least 5M tenge | 7,466 | **16.13%** |
| Age 0–5 years | 3,382 | 16.91% |
| Age 6–10 years | 1,600 | 15.91% |
| Age 11–20 years | 3,408 | 18.60% |
| Age 21+ years | 4,249 | **29.54%** |

The intersection age 21+ and price below 5M contains 3,584 rows, has 31.00%
MAPE, and creates approximately 40.9% of all percentage error. Performance
claims should therefore always include segment metrics.

## Prediction intervals

The main interval is conformally calibrated on out-of-fold residuals and
targeted at approximately 80% coverage. Mondrian adjustments use predicted
price groups, which are available during inference. A coarse fixed range is
used only if the interval artifact is unavailable.

Coverage describes repeated held-out listing prices. It is not the probability
that one particular final transaction occurs inside the range.

## Known limitations

- The target is advertised price, not transaction price.
- Price-basis classification is rule-based and intentionally conservative;
  ambiguous wording is not automatically discarded.
- Training covers one city and should not be presented as a national model.
- Older inexpensive vehicles have much higher error.
- Detail-page text/options cover only a minority of rows.
- Comparable listings require a local PostgreSQL database and are absent from
  the stateless public demo.
- The live service does not inspect uploaded text or photos as model features.
- Market behavior and prices can drift after the training period.
- MAPE heavily penalizes absolute errors on inexpensive vehicles.

## Computer vision

CV is experimental and excluded from production inference. Historical
`damaged` annotations mixed impact with cosmetic conditions, so all 47 affected
legacy frames were quarantined before being reviewed under the corrected
definition. The full 784-frame journal now has no pending rows: 18 boxed
`damaged`, 6 `wreck`, 9 `parts`, 615 `intact`, and 136 `unclear`. This yields
only 18 independent damaged/wreck listings, so prior supervised CV metrics
remain withdrawn until substantially more positives and an independent audit
are available.

## Reproducibility

The metadata artifact records the target policy, feature schema, data
fingerprint, code hash, Git commit, dirty-worktree flag, training row count,
validation metrics, and routing contract. Reproduce the offline chain with:

```bash
source .venv/bin/activate
python -m kz.ops.run_all --ml
python -m kz.ops.pipeline_status
```

Tests and lint:

```bash
python -m pytest tests/ -q
ruff check kz/ tests/ airflow/
```

## Product gate

The 18% MAPE objective is a research gate, not a published promise. A defensible
release improvement requires:

1. better evidence for physical condition in the under-5M segment;
2. a statistically supported grouped and temporal improvement;
3. unchanged or improved calibration;
4. no train/serve skew;
5. a fresh model card and artifact metadata produced by the same run.
