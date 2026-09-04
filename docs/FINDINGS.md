# Verified findings and failed experiments

This document records negative results as first-class project output. A failed
experiment narrows the search space, protects future work from repetition, and
is more useful than an unexplained metric improvement.

Metrics below refer to the historical snapshot on which each experiment was
run. The current production numbers live in [MODEL_CARD.md](MODEL_CARD.md).
Withdrawn CV figures are retained only to explain why they must not be cited.

## Executive summary

| Question | Finding | Decision |
|---|---|---|
| Do more listing-table features lower MAPE? | Four feature groups added noise. | Keep the 13-feature contract. |
| Does seller text help? | Sparse full-data text hurt; fully enriched cheap rows showed a local gain. | Improve unbiased coverage and close train/serve skew first. |
| Do generic image embeddings improve price? | ResNet50 and full-frame CLIP did not beat tabular baselines. | Do not add them to production price inference. |
| Does more same-source listing data solve the error? | Repeated growth moved MAPE little and the baseline caught up. | Treat listing-card data as plateaued. |
| Where is the error? | Old vehicles below 5M tenge dominate. | Prioritize physical-condition evidence. |
| Is the anomaly detector a fraud classifier? | No confirmed fraud in the labelled sample. | Keep human review and random controls. |
| Is supervised photo damage ready? | No; definition drift invalidated legacy positives. | Quarantine and relabel before evaluation. |

## 1. Additional tabular feature groups did not help

Four groups were tested beyond the stable 13-feature schema: richer technical
attributes, publication metadata, interaction-style transformations, and
enrichment-derived fields. They either left grouped MAPE unchanged or made it
worse. Sparse categories increased variance and coverage patterns encoded the
pipeline's own selection policy.

Decision: keep a compact feature contract unless a new field has adequate
coverage, exists at inference, and improves grouped plus temporal validation.

## 2. Image embeddings failed for different reasons

### 2.1 ResNet50 embeddings focused on the wrong signal

Generic ImageNet features mainly represented scene, angle, colour, and vehicle
type. They added dimensionality without useful price-condition information and
did not lower duplicate-safe validation error.

### 2.2 Cover-photo CLIP asked the wrong frame

The first image is usually selected to sell the car, not reveal its defect.
Encoding only the cover therefore tested marketing-photo composition rather
than vehicle condition.

### 2.3 All-frame CLIP found signal but not incremental value

Aggregating the gallery made age and broad condition more visible, but the
features did not improve price estimates beyond age and price-related tabular
information. A model can detect a visual pattern without adding product value.

Decision: full-frame embeddings are not production price features. Localized
damage detection remains a separate, narrower research question.

## 3. Target-derived features are forbidden

Kolesa's reference average price and price category make validation look better
because they derive from the marketplace target. They are useful for anomaly
cross-checks and stratification, never for training the price estimator.

## 4. Repeated-digit mileage was not a reliable fraud rule

Rounded or repeated mileage values looked suspicious by intuition, but review
did not show sufficient association with deception. Treat them as ordinary data
quality, not fraud evidence.

## 5. The current ceiling is a signal problem

At 5M tenge and above, MAPE is about 16%. Below 5M it is about 29%. The largest
intersection is age 21+ and price below 5M. Listing-table data describe vehicle
identity well but not corrosion, crash repair, mechanical state, or restoration
quality. Hyperparameter tuning cannot recover a signal that is absent.

## 6. Zero confirmed fraud is still a measurement

The manual sample currently contains no confirmed fraud. This does not prove
zero prevalence. Sixty-five random controls with zero positives imply a rough
one-sided 95% upper bound of `3 / 65 ≈ 4.6%` by the rule of three.

Decision: describe candidates as anomalies, retain `unknown`, and continue
random-control review before claiming recall or prevalence.

## 7. More data of the same type reached a plateau

Several collection rounds substantially increased rows but moved grouped MAPE
by tenths or hundredths of a percentage point. One larger growth stage improved
MAPE by about 1.06 points, but later rounds produced almost no stable gain.

The simple make/model/year baseline improved faster as the market grid became
denser. By median APE it approached the CatBoost model. The ML model's remaining
advantage is mainly in the tail.

Decision: fresh collection remains useful for drift and market coverage, but
deep pagination of the same fields is not the path to 18% MAPE.

## 8. Changing the loss was not a free improvement

Training closer to absolute error improved the median case but worsened average
percentage error. That is a trade-off, not a universal improvement. The chosen
objective must match the product metric and segment priorities.

## 9. A global multiplier improved MAPE by adding bias

Multiplying all predictions by approximately 0.95 reduced MAPE by about 0.52
percentage points on one evaluation, but systematically underpriced vehicles.
This exploits asymmetry in percentage error rather than learning a better fair
price.

Decision: reject metric gaming that worsens product calibration.

## 10. Interval tail imbalance was a conditioning artifact

When grouped by actual price, cheap vehicles appeared more often below the
lower bound than above the upper bound. This is expected: selecting rows with a
low actual target preferentially selects cases the model overpredicted.

Grouping by predicted price—the quantity available at inference—produced
balanced tails. The interval implementation was not defective.

## 11. Data growth helps, but the baseline also benefits

The model improved as the dataset grew, yet the simple group baseline closed the
gap faster for typical rows. “More data helps” and “the model's relative value
shrinks” can both be true.

Population Stability Index checks showed no major covariate shift during the
relevant window. The plateau was not explained by a dramatic population change.

## 12. Enrichment is valuable for anomaly exculpation

Detail-page context removed roughly 40% of some rule-positive suspicions by
revealing disclosed damage, non-running badges, instalment terms, or seller
explanations. This is a real product improvement even when price MAPE does not
move.

A later bug fix ensured that exculpation applies to residual-model candidates as
well as rule candidates.

## 13. The first photo conclusion was too broad

“Photos do not help” was incorrect. The actual result was narrower: full-frame
image features did not add price value beyond age and price-related features.
Separate zero-shot axes detected rust and dirt well in historical evaluation.

A proposed claim that better photos cause more views failed because the
observational data could not separate photo quality from vehicle and seller
confounders. No seller advice should promise more views from that test.

## 14. Geographic expansion was rejected

Adding Astana, Shymkent, or Karaganda would mix different demand, income,
logistics, and price regimes. With only a few thousand rows per city, a city
feature would mostly learn offsets while sparse make/model/year/city cells
remain weak.

The Almaty listing pool was also not exhausted. Expansion would change the
product from an Almaty estimator to an under-specified national estimator.

Decision: keep one city until a separate multi-market design is justified.

## 15. The objective was reframed after the plateau

The old intuition that roughly 17,000 similar rows would automatically reach
18% MAPE was unsupported. The measured target became segment-specific:

```text
vehicles at 5M+ KZT: keep MAPE below 17%    achieved at about 16%
vehicles below 5M:   reduce ~29% toward 20.5%
```

If the stronger segment remains unchanged, the latter improvement is roughly
what the weighted arithmetic requires for 18% overall.

## 16. Tiling showed that impact damage is local

Historical zero-shot damage AUC increased from 0.776 on whole frames to 0.827
when the maximum tile score was used. Rust moved in the opposite direction,
from 0.881 to 0.809. Dents are local; rust often covers a larger body area.

Decision: collect localized impact boxes and keep rust as a separate signal.

## 17. Enrichment can help price in the right subset

An early measurement incorrectly concluded that enriched fields had no value
because it compared mismatched samples. On a fully enriched cheap subset,
seller text and options improved the local result by about 3.4 percentage
points. On the full dataset with only about 12% useful enrichment coverage, the
gain shrank to about 0.05 points.

Coverage was also non-random: selected suspicious rows had shorter and
different text. Presence of enrichment could therefore encode pipeline policy.

Decision: improve balanced coverage and expose the same inputs at serving time
before adding these features.

The detail-page audit also showed that structured `page_condition` is not always
present and public pages do not reveal the VIN. The parser now stores only
positive evidence for a vehicle-history/VIN-backed flag and never stores VIN.

## 18. A fifth measurement confirmed the plateau

An approximately 18% increase in training data moved overall MAPE by about 0.19
percentage points while median error worsened. The baseline's median approached
the model's median, reinforcing that CatBoost's value was concentrated in hard
tails rather than typical rows.

## 19. A supervised damage score looked strong for the wrong reason

An early classifier produced an attractive overall AUC, but age plus price alone
explained much of it: damaged vehicles in the labelled sample were simply older
and cheaper. Inside the inexpensive segment, image performance did not clearly
beat the tabular baseline.

Lesson: compare every CV model with a strong non-image baseline and evaluate by
independent listings, not frames.

## 20. Bounding-box crops helped but did not clear the gate

On the historical labels, whole-frame CLIP in the cheap segment had AUC around
0.607, tile/crop aggregation around 0.633, and age plus price around 0.704.
The no-body axis performed strongly on manually identified interiors, supporting
queue ordering rather than exclusion.

An annotator also found that boxes drawn around rust were silently discarded
when the frame label was `intact`. The journal now preserves boxes with every
label and requires them only for `damaged`.

All figures in this section are historical and not current claims because the
positive labels were later quarantined for definition drift.

## 21. Part of cheap-segment MAPE is metric arithmetic and target noise

The same absolute tenge miss creates a larger percentage error on a cheap car.
Advertised prices also contain negotiation margins, missing condition, and
occasional non-comparable terms. This creates an irreducible floor unless the
target or evidence improves.

A separate cheap-segment model improved overall MAPE by roughly 0.25 percentage
points in its first honest experiment—useful but far from the required gain.

## 22. The cheap specialist became a valid production route

The first experiment routed by actual price, which is unavailable in production
and therefore invalid. The corrected route uses the general model's prediction
below 5M and trains the specialist on a wider actual-price band below 8M.

The current snapshot shows only -0.03 percentage points grouped improvement,
with a confidence interval crossing zero. The route is leakage-safe, but its
overall benefit is not statistically established on the latest data.

## 23. Reproducible full-frame CV remained negative

Photo evaluation was rebuilt with grouped folds, one prediction per listing,
paired bootstrap, and both ROC-AUC and PR-AUC. CLIP did not establish an
incremental benefit over age plus price. PCA was also moved inside each fold to
prevent distribution leakage from the test fold.

## 24. Active learning cannot define the final test set

Model-ranked frames are intentionally enriched for likely positives. Treating
them as a test set would overstate real-world prevalence and entangle evaluation
with the current model. New listings now receive a deterministic random audit
split before ranking. Legacy labels remain training-only because they already
influenced experiments.

Exact-photo pHash components are grouped in addition to `ad_id` so copied images
cannot cross train/test boundaries under different listings.

## 25. Another 461 rows moved MAPE by only 0.13 points

A later collection round added 461 training rows. Overall MAPE improved from
about 21.52% to 21.39%, while the under-5M segment barely changed. This is within
the broader plateau pattern and does not justify more deep pagination as the
main strategy.

## 26. The hard segment is not “all vehicles older than five”

Measured age bands showed approximately 16% MAPE for 6–10 years, 18% for 11–20,
and about 30% for 21+. The age-21+-and-below-5M intersection represented about
28% of rows but approximately 41% of all percentage error.

Decision: target that intersection rather than describing every vehicle older
than five years as equally difficult.

## 27. Another 149 rows left overall MAPE unchanged

A fresh block increased training rows from 11,991 to 12,140. Routed grouped MAPE
moved from 21.3927% to 21.3951%, effectively zero; median APE worsened by about
0.11 points. The cheap segment improved slightly, while out-of-time MAPE worsened.

One positive signal appeared: routed inference beat the general model on that
temporal holdout with a paired confidence interval below zero. Later snapshots
did not preserve a conclusive overall advantage, so both grouped and temporal
evidence remain necessary.

## 28. Definition drift invalidated supervised photo claims

The old interface used a broad term that annotators reasonably interpreted as
including rust, scuffs, dirt, and paint defects. Comments on all 47 legacy
`damaged` frames showed that 38 required visual re-review; only a small subset
clearly described impact, dents, deformation, or missing parts.

Every legacy `damaged` row was marked `needs_review` non-destructively. The
original CSV was backed up, pending rows are excluded from training and COCO
export, and no automatic relabeling was attempted. Only three independent
positive listings are currently verified for CV.

Decision: withdraw all supervised CV figures until the labels are reviewed
under the exact English protocol.

## 29. Targeted enrichment changed anomaly flags, not MAPE

A 2 September enrichment batch added 20 detail pages and 20 average-price/badge
records without HTTP 429. Six rule alerts were exculpated. After the full ML
chain, grouped MAPE moved by +0.0439 percentage points—far below the roughly
0.25-point bootstrap standard deviation.

This was not a meaningful degradation. Enrichment improved anomaly evidence;
the batch was simply too small and too sparsely covered to move global price
accuracy.

## 30. The listing number needs an explicit price-basis policy

One enriched listing advertised 7.0M KZT without customs clearance, 10.9M KZT
with customs clearance, and 11.4M KZT on credit. The saved listing target was
7.0M. Treating a generic credit or customs keyword as a row-level flag would be
wrong because all three meanings occur in the same description.

A contextual classifier now parses amounts and associates the saved price with
the nearest supported cue. It also handles customs negation, spelling variants,
clause boundaries, and disagreement between structured fields and prose.
Ordinary dealer finance boilerplate did produce false positives in the first
draft; corpus review caught them, and credit/down-payment labels now require the
advertised amount to be explicitly tied to the cue.

On the corpus audit, 26 of 12,799 rows were classified as `cash_uncleared`; 24
had previously been eligible for training. No current row met the
high-confidence credit-price or down-payment rule. Ambiguous rows remain
eligible.

A controlled grouped-CV A/B on the same snapshot measured **21.7327% MAPE
without** this filter and **21.6333% with** it, an improvement of 0.0993
percentage points. That small change is below the model's overall bootstrap
variation, but the target definition is more correct independently of the
headline metric. The previous artifact's 21.3044% is not a valid A/B baseline:
manual verdicts changed the training cohort between those runs.

The same eligibility rule now governs model training, floor calibration,
residual review, generated reports, CLI examples, and local comparable listings.
This closed a train/report skew found when the first updated dashboard still
reported 12,666 rows instead of the artifact's 12,642.

## Practical rules derived from these findings

1. Measure on grouped OOF and out-of-time predictions, never training rows.
2. Report uncertainty before interpreting a change of a few hundredths.
3. Compare images with age-plus-price, not with a coin flip.
4. Keep active learning separate from the random audit.
5. Never mix rust, cosmetic wear, and impact under one damage class.
6. Do not add a feature until it exists at both train and serve time.
7. Do not call an anomaly fraud before a human verdict.
8. Do not expand geography without redefining and validating the product.
9. Prefer new condition evidence over more repetitions of plateaued fields.
10. Preserve failed experiments so future work starts from evidence.
11. Classify what a displayed price means before treating it as a regression target.
