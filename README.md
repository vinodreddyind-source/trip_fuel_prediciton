# Trip Fuel Prediction — MLOps Pipeline (Practice Project)

A practice implementation of an airline trip-fuel prediction pipeline: a model
predicts expected fuel burn per flight, flags flights that deviate
meaningfully from that prediction, and includes the full MLOps lifecycle
around it — data versioning, model versioning, drift detection, and
gated, scheduled retraining via CI/CD.

> **Note on data**: this project uses synthetic flight data, not a real
> airline's records. It's a portfolio/practice build demonstrating the same
> MLOps patterns (baseline-gated training, containerized serving, drift
> monitoring, automated retraining) used on a real production system I built
> previously with genuine airline data.

## Architecture

**1. Training** (`train.py`) — time-based split using a **rolling window**
(most recent N months only, not all history), trains an XGBoost regressor,
and gates deployment against a historical-average baseline. Every run logs
to **MLflow** (params, metrics, model artifact to S3); a model is only
registered and promoted to the `production` alias if it beats the baseline.

**2. Evaluation** (`evaluate_savings.py`) — flags flights where actual fuel
exceeded the model's prediction by more than 5%, expressed as a percentage
of total fuel burned. A **backtested potential**, not a measured outcome
from flights that were actually re-planned.

**3. Data versioning** — **DVC**, with **S3** as the remote. The `data/`
folder is entirely DVC-managed (not git-tracked); `data.dvc` is the small
pointer file git actually tracks, so any commit can reconstruct the exact
dataset used for that training run via `dvc pull`.

**4. Drift detection** (`detect_drift.py`) — three distinct checks against
the current MLflow `production`-aliased model:
- **Aggregate data drift** (Evidently) — whole-column feature distribution
  comparison
- **Route-stratified data drift** — catches localized shifts (e.g. one
  route's weather pattern changing) that aggregate checks can dilute away
  when the shift only affects a small subset of the data
- **Concept/residual drift** — monitors the production model's actual
  prediction error against its own backtest baseline, catching cases where
  the *relationship* between inputs and output has changed (e.g. a fleet
  retrofit), even when input distributions look unchanged

**5. Scheduled retraining** (`.github/workflows/mlops_pipeline.yml`) — runs
weekly (plus manual trigger). If drift is flagged, it incorporates new data,
retrains, and — only if the new model beats the baseline gate — pushes the
updated data version and model back to the repo automatically.

**6. Serving** — FastAPI, containerized with Docker, tested locally; see
[Deployment](#deployment) below for the AWS Lambda + ECR path.

## Results (from an actual run against the synthetic dataset)

| Metric | Value |
|---|---|
| Baseline (historical average) MAE | 477 kg |
| Model (XGBoost) MAE | 354 kg |
| Improvement over baseline | 25.7-25.8% |
| Backtested potential fuel savings | 0.86-0.87% of total fuel burned |

**After a drift-triggered retrain** (fleet-retrofit scenario, engineered for
testing): improvement over baseline drops to a modest **4.2%** on the new
data — a realistic result, not a dramatic fix. Only ~4% of the training
window (after incorporating new data) reflects the new regime, since a
single retraining cycle right after a shift only partially closes the gap.
This is a real, well-known MLOps pattern: retraining improves incrementally
as more post-shift data accumulates over successive cycles, which is exactly
why continuous/scheduled retraining exists rather than a one-time fix.

## Running it

```bash
pip install -r requirements.txt
python data/generate_data.py       # regenerate synthetic data (fixed seed - reproducible)
python train.py                    # train + backtest gate + MLflow logging
python evaluate_savings.py         # worked savings calculation
```

## Drift detection and retraining

```bash
python data/generate_week2_data.py   # generates an engineered drift scenario
python detect_drift.py               # runs all three drift checks
python incorporate_new_data.py       # merges new data into the training set
python train.py                      # retrain, re-gate, re-promote if it passes
```

`detect_drift.py` exits with code `1` if any drift check is flagged, `0`
otherwise — used directly by the CI workflow to decide whether to retrain.

## Data and model versioning

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs, metrics, registry
aws s3 ls s3://trip-fuel-mlops-vinod/dvc-store/ --recursive        # versioned data
aws s3 ls s3://trip-fuel-mlops-vinod/mlflow-artifacts/ --recursive # versioned models
```

## CI/CD

`.github/workflows/mlops_pipeline.yml` runs on a weekly schedule and on
manual dispatch. Requires two repository secrets: `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, scoped to an IAM user with S3 access to the
project's bucket.

## Running it with Docker

```bash
docker build -t trip-fuel-api .
docker run -p 8000:8000 trip-fuel-api
```

Then visit `http://localhost:8000/docs` for an interactive Swagger UI.

## Deployment

Deployed to AWS as a container-image Lambda function behind a Function URL
— see `lambda/` for the Lambda-specific handler and Dockerfile, and
[below](#aws-deployment-notes) for the deployment steps.

## What this demonstrates

- Baseline-gated model deployment (a model must beat a simple lookup before
  being considered a valid replacement)
- Rolling-window retraining, chosen specifically to avoid a real failure
  mode: a fixed historical training window dilutes a recent regime shift
  into near-invisibility as more history accumulates
- Data and model versioning with genuine lineage (any past run's exact
  data and model can be reconstructed)
- Two distinct drift-detection strategies (data vs. concept), because a
  model can look fine on input distributions while being systematically
  wrong after a real-world change
- Scheduled, gated retraining via CI/CD — new data doesn't get promoted to
  production unless it actually improves on the baseline
- Containerized serving, both for local Docker and AWS Lambda

## Lessons learned building this (kept deliberately, not polished away)

- A model-tiering choice (rolling training window) was needed after
  discovering that a percentile-based time split silently excluded 100% of
  newly-incorporated data from training — the retrain looked successful
  while learning nothing new
- A first attempt at route-level drift detection under-triggered because
  the *reference* data's per-route variance didn't reflect real route-specific
  patterns — fixed with a documented, deliberately lower significance
  threshold appropriate to the sample size
- CI debugging: a Python version mismatch between the tested environment
  and the Docker base image, a missing executable in a multi-stage Docker
  build, a `pip`-installed CLI tool not reliably on `PATH` in GitHub
  Actions, and a data folder accidentally double-tracked by both git and
  DVC — all real, all fixed, all traceable in the commit history
