# Market anomalies and data cleaning

## The core definition

In this project, **fraud means deception**, not an unattractive or damaged
vehicle. An honestly described wreck at a low price is a legitimate listing.
An unusually high price can be a poor deal without being fraud.

The anomaly system therefore does not auto-ban sellers. It protects the price
training set and creates a human-review queue.

## Cleaning and screening flow

### 1. Normalize types

Prices, mileage, years, engine displacement, counts, and Boolean indicators are
converted explicitly. Impossible values are rejected or marked missing. Model
categories remain in their original marketplace vocabulary because those exact
values define the trained feature space.

### 2. Preserve missingness

A missing value is not silently converted to a plausible zero. Dedicated flags
such as `mileage_missing` let the model distinguish unknown mileage from a real
zero. Enrichment absence is treated carefully because it may reflect queue
selection rather than seller behavior.

### 3. Apply deterministic rules

Current rules identify candidates such as:

- price far below a make/model/year reference;
- a recent vehicle at an implausibly low price;
- a used vehicle recorded with zero mileage;
- possible relisting with conflicting attributes;
- an exact photo reused across different vehicles;
- low price combined with urgency language.

These are candidate-generating signals, not final verdicts.

### 4. Work in log-price space

Vehicle prices span a wide range. The model and many residual checks use
`log(price)` so that a multiplicative discrepancy has similar meaning across
cheap and expensive segments.

### 5. Use robust deviation

For a value `x`, median `m`, and median absolute deviation `MAD`, the modified
z-score is:

```text
z = 0.6745 × (x - m) / MAD
```

Median-based scale is less sensitive to extreme advertisements than mean and
standard deviation. It helps rank unusual prices within comparable groups.

### 6. Parse damage text with negation

Damage keywords are not matched blindly. Phrases that deny damage must not be
treated the same as a positive disclosure. The parser retains original market
text because translating source evidence could change negation and meaning.

### 7. Exculpate explained low prices

A low price is less suspicious when seller text, photos, or a site badge clearly
discloses crash damage, a non-running vehicle, missing components, corrosion, or
an instalment down payment. Enrichment exists partly to collect this context.

The exculpation layer is applied to both rule and model candidates. A previous
bug cleared rule alerts but left an explained listing flagged by the residual
detector; tests now enforce consistent behavior.

### 8. Group relists

The same physical vehicle may reappear under a new ad ID. Grouping uses vehicle
attributes and text evidence but deliberately excludes price, because price
changes are often the phenomenon under investigation. Entire groups stay in one
validation fold.

### 9. Detect exact photo reuse

Perceptual hashes identify exact or effectively exact photo copies. The current
production threshold is Hamming distance zero: looser thresholds created false
pairs among common dealer-style studio images. Photo reuse is evidence for
review, not proof of deception.

### 10. Build the final candidate flag

The clean layer combines rules, unexplained residuals, relist/photo evidence,
and any existing manual verdict. A final human `legit` verdict must remain valid
even if a row is still unusual numerically.

## Human review protocol

Run the single local application:

```bash
python -m kz.web
```

Open `http://127.0.0.1:8000/label`. The queue combines:

- `rule_positive`: deterministic flags;
- `residual_candidate`: prices unexpectedly low for the model;
- `random_control`: unflagged rows required to estimate misses.

Verdicts:

| Verdict | Meaning |
|---|---|
| `fraud` | Evidence supports deliberate deception or bait |
| `legit` | The listing is unusual but honestly explained or ordinary |
| `unknown` | Evidence is insufficient for a defensible decision |

The journal at `data/manual_labels.csv` is manual ground truth. It is never
recreated from a queue, never deleted as a whole, and is written atomically.
Repeated review updates the existing row.

## Why random controls matter

Reviewing only flagged rows estimates precision but cannot estimate recall.
Random controls reveal fraud the detector did not flag.

With weighted sampling, population metrics must use stratum weights. If stratum
`h` has population size `N_h`, reviewed sample size `n_h`, and `y_i` is a fraud
indicator, the estimated population total is:

```text
T_hat = Σ_h (N_h / n_h) × Σ_{i in h} y_i
```

The project stores sampling stratum in the durable journal because a rebuilt
queue intentionally removes completed work. Losing the stratum would make the
control sample unusable for unbiased evaluation.

## Current evidence

The labelled anomaly sample currently contains no confirmed fraud. That does
not prove the market has none. A random sample of 65 controls with zero observed
fraud gives a one-sided 95% upper prevalence bound of roughly 4.6% by the rule
of three (`3 / n`). The correct conclusion is limited: no fraud was found in
that sample, and larger review coverage is required for a tighter bound.

## Safety boundary

The public demo disables `/label`, `/damage`, `/verdict`, and photo-label writes.
Anonymous access to these routes would allow anyone to corrupt the project's
only ground truth. UI tests must use `KZ_LABELS_DIR` with a scratch directory.
