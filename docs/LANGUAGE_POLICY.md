# Language policy

The project's public language is English.

This rule covers:

- the live web application and its validation messages;
- the anomaly-review and photo-labeling interfaces;
- generated HTML reports and chart labels;
- command-line output, logs, and exceptions;
- maintained public Markdown documentation;
- code comments and docstrings.

## Why some Cyrillic strings remain in the source

Kolesa.kz publishes several categorical values and HTML markers in Russian.
Those strings are source data, not interface copy. Examples include the raw
tokens for petrol, automatic transmission, sedan, used condition, archive
status, and seller-language damage expressions.

They remain unchanged in four narrowly defined places:

1. parser dictionaries and regular expressions that read the marketplace;
2. cleaning rules that normalize historical source values;
3. test fixtures that reproduce real source pages and seller text;
4. model-input values stored in the trained artifact's categorical vocabulary.

Translating these internal values would change the feature distribution seen by
CatBoost and could silently break parsing, validation, or inference.

## Display values versus model values

The estimator follows a deliberate two-layer contract:

```text
English label shown to the user  →  original source value submitted to the model
Petrol                           →  бензин
Automatic                        →  автомат
Sedan                            →  седан
Used                             →  б/у
```

The same mapping is applied when raw categorical values appear in explanations
or generated reports. Users see English; the trained model receives the exact
vocabulary it learned during training.

## Adding a new category

When the marketplace introduces a new category:

1. preserve the exact source token in collection and normalization code;
2. add an English display label at the UI boundary;
3. add a parser fixture and an inference-contract test;
4. retrain only if the category enters the production feature vocabulary;
5. verify grouped and temporal metrics before publishing a new artifact.

This boundary keeps the product internationally readable without corrupting
source evidence or creating train-serving skew.
