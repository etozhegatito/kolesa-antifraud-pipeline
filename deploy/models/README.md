# Public demo model artifacts

This directory contains only the six trained CatBoost artifacts required by
the read-only price demo:

- `price_model.cbm` — general price model;
- `price_cheap_specialist.cbm` — specialist used after a predicted price below
  ₸5 million;
- `price_model.metadata.json` — features, routing contract, provenance hashes,
  target policy, and validation metrics.
- `price_interval_lower.cbm` and `price_interval_upper.cbm` — lower and upper
  quantile models;
- `price_interval.metadata.json` — grouped conformal offsets and measured
  coverage.

The artifacts were trained on 12,639 rows and created on 5 September 2026.
Their primary grouped out-of-fold MAPE is 21.48%, and the interval covers 80.1%
of grouped OOF listings against an 80% target. The target policy excludes known
uncleared-cash, credit-price, down-payment, and strict missing-powertrain parts
amounts while retaining ambiguous rows.

No source listing, seller description, image, URL, manual verdict, damage
label, contact detail, or database credential is stored here. CatBoost `.cbm`
files are derivative tree weights required for inference, not a browsable copy
of the marketplace dataset.

After an approved retraining run, replace all six files together and rerun the
Docker health smoke. Never update only one routed or interval artifact.
