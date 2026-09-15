# Wine Quality ML Platform

Predicting red wine quality from physicochemical measurements, following
Cortez et al. (2009). The dataset's stated goal is to *"use machine learning to
determine which physiochemical properties make a wine 'good'"* — this project
answers that with a served model, an interactive dashboard, and a reproducible
training pipeline.

**Dataset:** [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) ·
1,599 samples · 11 features · 1 target

---

## Quick start

```bash
# 1. Train the models (writes artifacts to models/)
make install
make train

# 2. Run the stack
make api          # terminal 1 — http://localhost:8000/docs
make dashboard    # terminal 2 — http://localhost:8501
```

Or entirely in Docker:

```bash
docker compose --profile train up --build   # trains, then serves
docker compose up                           # serve only (artifacts must exist)
```

| Service | URL | Purpose |
|---|---|---|
| Dashboard | http://localhost:8501 | EDA, model performance, live predictions |
| API docs | http://localhost:8000/docs | Interactive OpenAPI reference |
| API health | http://localhost:8000/health | Liveness and model-readiness |

---

## What this project does

The dataset supports two framings, and this project implements **both**:

| Task | Target | Model | Primary metric |
|---|---|---|---|
| **Classification** | `is_good` — quality ≥ 7 | Best of 3 candidates | **PR-AUC** |
| **Regression** | `quality` — score 3–8 | Best of 3 candidates | RMSE |

### Why PR-AUC is the headline metric

Only **13.6%** of wines clear the quality ≥ 7 cutoff — a 6.4:1 imbalance. A model
that predicts "never good" scores **86.4% accuracy while finding zero good
wines**. Accuracy and ROC-AUC both hide this failure; average precision does not.
Training therefore uses `class_weight="balanced"` rather than synthesising rows
with SMOTE, and the classifier is ranked by cross-validated PR-AUC.

### What makes a wine good?

The trained classifier's feature importances answer the dataset's inspiration
question directly. `alcohol` is consistently the strongest signal, followed by
`volatile acidity` and `sulphates` — matching the domain literature. These are
**correlations, not causes**; the dataset card explicitly warns against causal
readings.

---

## Architecture

```
raw_data/winequality-red.csv
        │
        ▼
┌───────────────────┐   validate schema + ranges, derive both targets
│   Data layer      │   src/wine_quality/data/
└────────┬──────────┘
         ▼
┌───────────────────┐   ONE preprocessing definition, shared by train & serve
│  Feature layer    │   src/wine_quality/features/
└────────┬──────────┘
         ▼
┌───────────────────┐   cross-validated selection, serialised pipelines
│   Model layer     │   src/wine_quality/models/
└────────┬──────────┘
         ▼  models/classifier.joblib · regressor.joblib · model_metadata.json
┌───────────────────┐   FastAPI, loads artifacts once at startup
│  Serving layer    │   src/wine_quality/serving/
└────────┬──────────┘
         ▼  HTTP
┌───────────────────┐   Streamlit, 4 pages, no in-process model import
│ Presentation      │   app/
└───────────────────┘
```

The single most important design decision: **preprocessing lives inside the
serialised pipeline**. The `ColumnTransformer` is fitted during training and
saved *with* the estimator, so inference replays the exact transformation the
model was trained on. Train/serve skew becomes structurally impossible rather
than merely discouraged.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design, including data
contracts, API schemas, and the phased build roadmap.

---

## Project layout

```
├── raw_data/                    # Source CSV (never modified)
├── src/wine_quality/
│   ├── config.py                # All settings, feature schema, bounds
│   ├── data/                    # Loading, validation, target derivation
│   ├── features/                # The shared preprocessing contract
│   ├── models/                  # Candidates, training, evaluation, registry
│   └── serving/                 # FastAPI app and wire schemas
├── app/                         # Streamlit dashboard
│   ├── api_client.py            # HTTP client (no in-process imports)
│   ├── Overview.py              # Landing page
│   └── pages/                   # Explore · Performance · Predict
├── tests/                       # 4 tiers: data, features, models, API
├── models/                      # Generated artifacts (gitignored)
├── reports/                     # metrics.json + runs.jsonl (gitignored)
├── Dockerfile                   # Multi-stage: base → builder → runtime → training
├── docker-compose.yml           # api + dashboard + profile-gated train
└── Makefile                     # make help
```

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether models are loaded |
| `GET` | `/model-info` | Served model's provenance and held-out metrics |
| `GET` | `/schema` | Feature names, ranges, descriptions |
| `POST` | `/predict` | Both tasks in one call |
| `POST` | `/predict/classification` | Binary verdict only |
| `POST` | `/predict/regression` | Quality score only |
| `POST` | `/reload` | Re-read artifacts without a restart |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4
  }'
```

```json
{
  "classification": {
    "is_good": false,
    "probability_good": 0.2134,
    "threshold": 0.5,
    "model_version": "0.1.0"
  },
  "regression": {
    "predicted_quality": 5.42,
    "rounded_quality": 5,
    "model_version": "0.1.0"
  },
  "input_features": { "...": "echoed for traceability" }
}
```

Note the field names use underscores (`fixed_acidity`) while the CSV uses spaces
(`fixed acidity`). The API accepts the underscore form; the mapping is handled
internally.

---

## Configuration

Every setting is overridable via environment variables with the `WINE_` prefix.

| Variable | Default | Purpose |
|---|---|---|
| `WINE_GOOD_QUALITY_CUTOFF` | `7` | Score at or above which a wine is "good" |
| `WINE_TEST_SIZE` | `0.2` | Held-out fraction |
| `WINE_RANDOM_STATE` | `42` | Seed for splits and estimators |
| `WINE_CV_FOLDS` | `5` | Cross-validation folds |
| `WINE_CLASS_WEIGHT` | `balanced` | Imbalance mitigation |
| `WINE_API_BASE_URL` | `http://localhost:8000` | Dashboard → API URL |
| `WINE_API_PORT` | `8000` | API port |
| `DASHBOARD_PORT` | `8501` | Dashboard port |

```bash
# Try a stricter definition of "good"
WINE_GOOD_QUALITY_CUTOFF=8 make train
```

---

## Testing

```bash
make test        # full suite with coverage
make test-fast   # skip the slow end-to-end training tests
make lint        # ruff
```

Four tiers, each targeting a different failure mode:

| File | Guards against |
|---|---|
| `test_data.py` | Schema drift, corrupted CSVs, wrong target derivation |
| `test_features.py` | Preprocessing changes that would silently alter production |
| `test_models.py` | Metric bugs, artifact round-trip failures, weak models |
| `test_api.py` | Contract violations, validation gaps, non-determinism |

`test_features.py` is the most important file: because the preprocessor is
serialised into the model, any change there changes what production computes.
Those tests pin the behaviour explicitly.

---

## Deployment

The compose setup above is the **development** topology: two services that can
be restarted independently. Public hosts that publish a single port need a
different arrangement, so `deploy/huggingface/` packages the same two services
into one container behind nginx.

```
browser
   │
   ▼   :7860  (the only published port)
──────────────┐
│    nginx     │
└──┬────────┬──┘
   │        │
   │  /api/ │  /  (everything else)
   ▼        ▼
 FastAPI  Streamlit        both on loopback, 127.0.0.1
 :8000     :7861
```

The dashboard still reaches the API over HTTP rather than importing the models,
so the service boundary survives the merge — the deployed topology has the same
shape as the compose one, not a different architecture.

### Why a separate repository

Hugging Face only auto-detects a Dockerfile named `Dockerfile` at the repository
root, and its README frontmatter must sit there too. Neither can be relocated by
configuration, so the Space is assembled as its own repository by
`scripts/deploy_hf_space.sh`. The main project keeps its two-service compose
setup untouched.

### Deploying

```bash
pip install -U "huggingface_hub[cli]"
hf auth login

make train                        # artifacts are baked into the image
./scripts/deploy_hf_space.sh <hf-username>/<space-name>
```

The script refuses to deploy without model artifacts. A Space has no compose
volume, so the ~16 MB of `.joblib` files are baked into the image; shipping
without them would start the API in its degraded state and return 503 on every
prediction.

### Platform constraints that shaped this

| Constraint | Consequence |
|---|---|
| One published port | nginx multiplexes `/` and `/api/` on 7860 |
| Container runs as UID 1000 | nginx `pid` and temp paths moved under `/tmp` |
| Build context has a size budget | `.dockerignore` keeps the context at ~17 MB |
| No compose volumes | Model artifacts are copied into the image |

`deploy/huggingface/entrypoint.py` supervises all three processes and tears the
container down if any one of them exits. That matters: with a bare `&` and
`wait`, a crashed API would leave nginx serving a dashboard whose every
prediction fails — a half-broken demo that is far harder to diagnose than a
failed deployment.

---

## Design decisions

**Why two models instead of one?** The score and the probability answer different
questions. The regressor places a wine on the original sensory scale; the
classifier expresses confidence that it clears the cutoff. They are fitted
separately and can disagree — that disagreement is informative, not a bug.

**Why not MLflow or DVC?** The dataset is 100 KB and static, and runs are
infrequent. An append-only `reports/runs.jsonl` that `git diff` can read is more
useful here than a tracking server. The trade-off is documented rather than
hidden.

**Why does the dashboard call the API instead of importing the model?** Going
over HTTP means the UI exercises the same interface any other client would. A
broken API surfaces as a visible dashboard error instead of being masked by a
working local import.

**Why is training behind a compose profile?** `docker compose up` should be fast
for demos. Training cross-validates six pipelines, so it is opt-in via
`--profile train` while remaining fully reproducible.

**Why keep the 240 duplicate rows?** They are genuine repeated measurements, not
data-entry errors. Removing them would discard real information; the effective
sample size is simply smaller than 1,599 suggests.

---

## Known limitations

- **Regression to the mean.** The regressor rarely predicts extreme scores (3 or
  8) because few such wines exist. Predictions cluster in the 5–6 range.
- **R² is modest (0.504).** Most quality variation is driven by factors this
  dataset does not capture — grape variety, vintage, winemaking technique.
- **No causal claims.** Feature importances are statistical associations.
- **Red wine only.** The white wine dataset is not included.
- **No authentication.** The API is intended for local or trusted-network use.

---

## Citation

```bibtex
@article{cortez2009modeling,
  title   = {Modeling wine preferences by data mining from physicochemical properties},
  author  = {Cortez, Paulo and Cerdeira, Ant{\'o}nio and Almeida, Fernando and Matos, Telmo and Reis, Jos{\'e}},
  journal = {Decision Support Systems},
  volume  = {47},
  number  = {4},
  pages   = {547--553},
  year    = {2009},
  doi     = {10.1016/j.dss.2009.05.016}
}
```

Dataset licensed under [ODbL / DbCL](http://opendatacommons.org/licenses/dbcl/1.0/).