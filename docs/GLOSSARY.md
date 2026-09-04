# Glossary

| Term | Meaning in this project |
|---|---|
| Ad / listing | A marketplace publication identified by an `ad_id`; one physical vehicle may have several listings. |
| APE | Absolute percentage error for one row: `|actual - prediction| / actual`. |
| MAPE | Mean APE across rows, multiplied by 100%. The primary price metric. |
| Median APE | Median row-level percentage error; describes a typical listing and resists extreme errors. |
| MAE | Mean absolute error, reported in the target's currency units. |
| R-squared | Fraction of variance explained relative to predicting the mean; calculated here in log-price space. |
| Baseline | A simple make/model/year reference used to prove that the ML model adds value. |
| Target | The value being predicted: first observed advertised price in KZT. |
| Price basis | What the displayed number represents: customs-cleared cash, uncleared cash, credit price, down payment, or ambiguous. |
| Feature | An input variable available to the model at training and inference. |
| Leakage | Information that would not legitimately be available at prediction time or crosses a validation boundary. |
| Train/serve skew | A mismatch between fields available during training and those supplied in production. |
| Log target | Modelling `log(price)` and converting predictions back with `exp`. |
| Out-of-fold (OOF) | Prediction for a row made by a fold model that did not train on that row's group. |
| Grouped CV | Cross-validation that keeps all relists of the same physical vehicle in one fold. |
| Out-of-time (OOT) | Evaluation on listings observed after the training period. |
| Bootstrap | Repeated resampling used to estimate metric uncertainty. This project samples relist groups. |
| Confidence interval | A range produced by repeated resampling; a paired delta interval crossing zero does not prove one model is better. |
| CatBoost | Gradient-boosted decision trees with strong support for categorical features. |
| SHAP | Additive contributions that explain how features moved one prediction from its expected value. |
| Conformal interval | A prediction range calibrated from held-out residuals to reach a target empirical coverage. |
| Coverage | Fraction of held-out actual values that fall inside predicted intervals. |
| Residual | Difference between an observed target and its prediction. |
| Quantile model | A model estimating a conditional percentile, used here to form a plausible lower price floor. |
| Anomaly candidate | A listing sent to human review; it is not automatically fraud. |
| Precision | Fraction of flagged/reviewed positives that are truly positive. |
| Recall | Fraction of all true positives found by the detector. Requires a control sample to estimate misses. |
| Fraud | Deliberate deception or bait, not merely a damaged or expensive vehicle. |
| Legit | A normal listing or an unusual price honestly explained by condition or terms. |
| Relist | The same physical vehicle published under a new ad ID. |
| Sighting | One observation of a listing on a particular day. |
| pHash | Perceptual image hash used to group exact photo copies. |
| CLIP | A pretrained image/text embedding model used in experimental photo analysis. |
| ROC-AUC | Ranking quality across all classification thresholds. |
| PR-AUC | Precision-recall area, especially informative for rare positive classes. |
| Bounding box | Relative `(x1, y1, x2, y2)` coordinates around a local image region. |
| Definition drift | Annotation meaning changing over time, making early and late labels incompatible. |
| Active learning | Prioritizing examples that a current model finds informative; these examples must not become the final random audit. |
| Right censoring | A listing remains active when observation ends, so its complete time-to-event is unknown. |
| PSI | Population Stability Index used to monitor feature-distribution change. |
| Idempotent | Safe to run again without creating duplicate effects. |
| Artifact | Saved model plus metadata required for reproducible inference. |
