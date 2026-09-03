# Public demo model artifacts

This directory contains only the three trained CatBoost artifacts required by
the read-only price demo:

- `price_model.cbm` — general price model;
- `price_cheap_specialist.cbm` — specialist used after a predicted price below
  ₸5 million;
- `price_model.metadata.json` — features, routing contract, provenance hashes,
  target policy, and validation metrics.

The artifacts were trained on 12,455 rows and created on 3 September 2026.
Their primary grouped out-of-fold MAPE is 21.36%.

No source listing, seller description, image, URL, manual verdict, damage
label, contact detail, or database credential is stored here. CatBoost `.cbm`
files are derivative tree weights required for inference, not a browsable copy
of the marketplace dataset.

After an approved retraining run, replace all three files together and rerun
the Docker health smoke. Never update only one routed-model artifact.
