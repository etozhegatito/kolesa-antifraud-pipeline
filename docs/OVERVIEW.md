# Project overview

## The problem

KZ Auto Market Intelligence estimates fair advertised prices for used vehicles
in Almaty. Marketplace data makes this harder than a normal regression exercise:

- the same vehicle may be removed and relisted under a new ID;
- prices can change while a listing remains active;
- low prices may be genuine bargains, disclosed crash damage, incomplete data,
  instalment down payments, or bait;
- the final transaction price is not observable;
- physical condition, which matters most for old inexpensive vehicles, is only
  weakly represented by structured listing fields.

The project treats those limitations as part of the product contract instead of
hiding them behind one accuracy number.

## Product scope

The primary product is a seller-facing listing-price estimator. Given make,
model, model year, mileage, engine, transmission, body style, condition, and
photo count, it returns:

1. a point estimate in Kazakhstani tenge;
2. a calibrated uncertainty range;
3. SHAP-based drivers of the individual estimate;
4. listing-quality and anomaly warnings;
5. comparable listings and market position when the local database is present.

Two internal review tools support the data system:

- `/label` assigns fraud, legit, or unknown verdicts to anomaly candidates;
- `/damage` assigns precise photo-condition labels and bounding boxes.

The public demo is read-only. It exposes estimation but disables both manual
labelling workflows because they modify irreplaceable ground truth.

## Meaning of “fair price”

The target is the **first observed advertised price**, not a completed-sale
price, dealer appraisal, repair-adjusted value, or guaranteed selling price.
The distinction matters: negotiation and final-sale data are unavailable.

The displayed number is not always the full cash price. The clean layer links
explicit amounts to customs, credit, and down-payment context and stores the
result as `price_basis`. Only confidently non-comparable targets are removed
from training; conflicting or missing evidence remains `ambiguous`.

A prediction range means that the calibration procedure covered approximately
the target fraction of held-out listing prices. It does not mean that a sale is
guaranteed inside the range.

## Current measured state

The current artifact was trained on 12,642 rows from 12,799 collected Almaty
listings.

| Validation view | Result |
|---|---:|
| Grouped out-of-fold MAPE, routed model | **21.63%** |
| Median absolute percentage error | **14.02%** |
| Out-of-time MAPE | **22.44%** |
| Grouped-bootstrap 95% interval for MAPE | **21.13%–22.17%** |
| Simple make/model/year baseline MAPE | **30.86%** |

The average error is not evenly distributed:

| Segment | MAPE |
|---|---:|
| Below 5M tenge | **29.45%** |
| 5M tenge and above | **16.21%** |
| Vehicle age 21+ years | **29.66%** |

Vehicles aged 21+ years and priced below 5M tenge create roughly 41% of the
total percentage error. The strongest practical path is therefore better
condition evidence, not more copies of the same listing-table fields.

## What is already reliable

- Raw advertisements and sightings are append-only evidence.
- The clean table is rebuilt deterministically from raw and enriched layers.
- Relists are grouped without using price in the grouping key.
- Grouped folds keep every copy of a vehicle on one side of validation.
- A later time block is held out for an out-of-time check.
- Price ranges are calibrated from held-out residuals.
- A specialist model is routed using the general model's prediction, never the
  unknown actual target.
- Anomaly flags create a review queue and do not accuse sellers automatically.
- CI covers offline tests, PostgreSQL integration, Ruff, coverage, and Docker
  model loading.

## Computer vision status

Computer vision is a research component, not a production claim. The repository
contains photo download, perceptual duplicate grouping, CLIP embeddings, tiled
analysis, manual bounding boxes, COCO export, grouped evaluation, and audit-split
logic.

An annotation audit found definition drift: the historical `damaged` class mixed
local impact damage with rust, dirt, scuffs, and paint defects. All 47 affected
legacy frames were marked `needs_review` without deletion or automatic
relabeling. Earlier supervised CV scores were withdrawn. The price model was not
affected because photo features never passed the product gate.

The next valid CV milestone requires 150–200 independent positive listings, a
random holdout selected before active learning, duplicate-component grouping,
and a positive paired-bootstrap improvement over the age-plus-price baseline.

## Outputs

The pipeline produces four classes of output:

1. `clean_data`: reproducible analytical rows with quality and anomaly fields;
2. trained general and cheap-segment CatBoost artifacts with metadata;
3. calibrated price intervals and anomaly thresholds;
4. HTML/JSON reports and protected human-review queues.

No raw listings, seller descriptions, photos, manual labels, credentials, or
private textbook files are shipped with the public repository.

## Roadmap

Work is prioritized by evidence:

1. improve detail-page enrichment coverage for inexpensive listings;
2. finish manual review under one stable photo-label policy;
3. train a true damage detector only after the annotation gate is met;
4. expose text and photo inputs only when training and serving use identical
   information;
5. repeat out-of-time validation after a longer market history;
6. keep 18% MAPE as a research gate, not a promise.

The intended future seller report should clearly separate observed facts,
model estimates, uncertainty, market comparisons, visual evidence, and warnings.
That separation is more valuable than presenting one overconfident number.
