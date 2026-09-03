# Price modelling and validation

## Current measured quality

The current artifact uses 12,455 training rows and reports:

| Evaluation | MAPE | Median APE |
|---|---:|---:|
| Grouped out-of-fold routed prediction | **21.36%** | **13.81%** |
| Out-of-time routed prediction | **23.24%** | **14.82%** |
| Grouped make/model/year baseline | 30.70% | 14.21% |

The 95% grouped-bootstrap interval for MAPE is 20.87%–21.87%. Changes smaller
than this sampling variation should not be described as real improvements.

## Target

The model predicts:

```text
y = log(first observed listing price in KZT)
```

The first observation is chosen explicitly so retraining does not silently
change the target from original price to latest price as observation history
grows. At inference, the output is transformed back:

```text
price_hat = exp(y_hat)
```

Log price reduces the dominance of very expensive vehicles and turns many
multiplicative price relationships into additive ones.

## Features

The deployed schema contains 13 features:

| Type | Features |
|---|---|
| Numeric | `age`, `mileage_km`, `engine_volume`, `photos_count` |
| Flags | `mileage_missing`, `is_vip`, `has_monthly_price` |
| Categorical | `brand`, `model`, `engine_type`, `transmission`, `body_type`, `condition` |

The marketplace's own average price and price category are forbidden as model
features because they derive from the same target. Seller text, options, and
photo features are not deployed until coverage and inference parity are strong
enough.

## Why CatBoost

CatBoost handles mixed numerical and high-cardinality categorical data without
manual one-hot expansion. Ordered target statistics reduce leakage compared
with naive target encoding, and tree interactions capture nonlinear effects
such as age behaving differently by make/model.

This is still compared with a simple baseline. A complex model has value only
if it beats a transparent reference under the same split.

## Baseline

The baseline predicts a robust central price for the make/model/year group,
with fallback levels when a group is sparse. Its grouped MAPE is much worse than
CatBoost, but median APE is close. This means much of the model's average gain
comes from reducing tail errors rather than improving every typical listing.

## Duplicate-safe grouped validation

A physical vehicle may be relisted under multiple ad IDs. Random row splitting
would let the model train on one copy and validate on another. The project first
constructs relist groups without using price and then uses grouped folds:

```text
all rows in one relist group → exactly one fold
```

Every row receives a prediction from a model that did not train on its group.
Those out-of-fold predictions drive model metrics, interval calibration,
residual anomaly thresholds, and stability analysis.

## Out-of-time validation

Grouped cross-validation estimates behavior across the observed dataset.
Out-of-time validation asks whether a model trained on earlier listings works on
later ones. The higher 23.24% temporal MAPE is a warning that market drift and
data-history limits remain important.

## Routed inference

One general model cannot fully capture the inexpensive segment. The route is:

```text
general prediction < 5M KZT → cheap specialist
otherwise                    → general model
```

The specialist trains on a wider band of actual prices below 8M KZT to reduce
edge instability. The route never uses actual price during inference.

On the current snapshot, routing changes overall grouped MAPE by only -0.03
percentage points and the paired confidence interval includes zero. The
architecture is correct, but the latest data do not support a strong claim of
overall improvement.

## Metrics

### Absolute percentage error and MAPE

For actual price `y_i` and prediction `p_i`:

```text
APE_i = |y_i - p_i| / y_i
MAPE  = (1 / n) × Σ_i APE_i × 100%
```

MAPE is intuitive but gives a fixed tenge error more weight on cheap vehicles.
That is why price-segment metrics and median APE are always reported.

### Median APE

```text
Median APE = median(APE_1, ..., APE_n) × 100%
```

The median describes a typical listing and is resistant to a small number of
large errors. A model can improve MAPE while leaving median APE unchanged if it
mainly fixes the tail.

### MAE

```text
MAE = (1 / n) × Σ_i |y_i - p_i|
```

MAE in tenge answers an important business question but naturally emphasizes
expensive vehicles. It complements rather than replaces MAPE.

### R-squared on log price

```text
R² = 1 - Σ_i (z_i - z_hat_i)² / Σ_i (z_i - mean(z))²
```

where `z = log(price)`. The current value is approximately 0.935. R-squared is
useful for fit diagnostics but less direct for seller-facing error.

## Confidence intervals

Rows are not independent within a relist group. Bootstrap resampling therefore
samples groups, not individual rows. Paired comparisons resample the same groups
for both models and compute the MAPE difference. A change is treated as
supported only when the confidence interval excludes zero in the intended
direction.

## Prediction ranges

The interval module calibrates absolute residuals on out-of-fold predictions.
The default target is roughly 80% coverage. Group-specific adjustments use
predicted price bands, because actual price is unavailable at inference.

Calibration answers a repeated-sample question: approximately what fraction of
similar held-out listing prices fell inside intervals produced by this method?
It does not create a probability guarantee for one transaction.

## Residual anomaly detector

A lower quantile model estimates how low a price can plausibly be for the
vehicle. Listings below the calibrated floor enter manual review unless their
low price is explained by explicit damage or another valid reason. The detector
is not a fraud classifier.

## Where the remaining error lives

Below-5M vehicles have 29.01% MAPE versus 16.07% above that threshold. The
21+-year, below-5M intersection alone contributes about 41.1% of total
percentage error. Repeated same-source data growth has reached a plateau.

The modelling roadmap therefore focuses on condition evidence and coverage,
not arbitrary hyperparameter tuning or unsupported feature expansion.
