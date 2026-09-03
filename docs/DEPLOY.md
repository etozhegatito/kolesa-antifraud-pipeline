# Deploying KZ Auto Market Intelligence

The public service exposes the price estimator only. Collection jobs and human
labeling workflows remain local because they access third-party pages or write
ground truth.

## What enters the image

| Asset | Included | Reason |
|---|---:|---|
| `kz/` source code | Yes | Required for inference and HTTP endpoints |
| General model, specialist, metadata | Yes | All three are required by routed inference |
| Raw listings and descriptions | **No** | Source marketplace content is not redistributed |
| Photos and photo URLs | **No** | Not required by the deployed estimator |
| Manual verdicts and damage labels | **No** | Private ground truth must remain protected |
| Playwright browser | **No** | Collection is not executed by the web service |

`requirements-web.txt` and `.dockerignore` keep the image focused on inference.

## Local container smoke test

```bash
docker build -t kz-auto-market-intelligence .
docker run --rm -p 8000:8000 \
  -e KZ_PUBLIC_DEMO=1 \
  kz-auto-market-intelligence
curl -s http://127.0.0.1:8000/api/health
```

The health endpoint loads both CatBoost models and validates their metadata,
so a successful response proves that the service can perform inference.

## Free Render deployment

The root `render.yaml` defines one Docker web service on Render's free plan:

- one instance and one Uvicorn worker;
- Frankfurt region;
- automatic deployment only after GitHub checks pass;
- `/api/health` model-loading health check;
- `KZ_PUBLIC_DEMO=1` to disable mutable labeling endpoints.

Create the service from the Blueprint:

1. open `https://dashboard.render.com/blueprints`;
2. connect `etozhegatito/kz-auto-market-intelligence`;
3. keep the Blueprint plan set to **Free**;
4. deploy and wait for the health check;
5. verify the public URL and `/api/health`.

Later pushes to `main` deploy automatically only after GitHub Actions passes.
No PostgreSQL service is needed for the public demo.

Free instances sleep after inactivity, so the first request can take about a
minute. The filesystem is ephemeral; this is safe because the public app is
read-only and the models are baked into the image.

## Public-mode boundary

`KZ_PUBLIC_DEMO=1` applies two safeguards:

- `/label`, `/damage`, and verdict-writing endpoints are unavailable;
- a per-address request limit protects against accidental loops.

This is a portfolio demo, not a hardened multi-tenant production API.

## Operation without PostgreSQL

Core price inference is artifact-only. If PostgreSQL is unavailable, the
service omits comparable-listing decorations while continuing to return the
estimate, calibrated range, explanations, and warnings. It logs the missing
database once instead of failing startup.

## Updating the model

Replace the three files under `deploy/models/` together, then run the tests and
Docker smoke before pushing. The metadata exposes the training timestamp,
training-row count, target policy, data fingerprint, code hash, and current
validation MAPE through `/api/health`.

Never publish raw `data/`, photos, manual labels, `.env`, generated reports, or
the private textbook.
