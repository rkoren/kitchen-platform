# Data Versioning with DVC

DVC is **opt-in**, not required. You can use `kitchen` without it — the
`kitchen run` commands work standalone. DVC adds input-hash caching so
unchanged stages are skipped, and an S3 remote so processed data and models
travel with the repo without being committed to Git.

Choose your path:

| | Without DVC | With DVC |
|---|---|---|
| Run pipeline | `kitchen run features && kitchen run train && kitchen run evaluate` | `dvc repro` |
| Skip unchanged stages | ✗ always re-runs | ✓ skips cached stages |
| Share data with teammates | manual S3 sync | `dvc push` / `dvc pull` |
| Reproduce exact results | re-run manually | `dvc repro` restores inputs then re-runs |
| Setup overhead | none | `pip install rkoren-kitchen[dvc]` + S3 bucket |

---

## Path A — Without DVC

No extra setup. Run stages directly:

```bash
kitchen ingest               # download raw data → data/raw/
kitchen run features         # build features → data/processed/
kitchen run train            # fit model → models/
kitchen run evaluate         # compute metrics → metrics.json
kitchen submit               # validate and upload Kaggle submission
```

Each command reads `menu.yaml` and writes outputs to the same paths every
time. Changing a hyperparameter and re-running replaces the previous output.

This path is sufficient for:
- Solo projects
- Kaggle competitions (raw data is re-downloadable by slug)
- Short experiments where reproducibility is tracked via MLflow alone

---

## Path B — With DVC

### New project

Pass `--with-dvc` to `kitchen init`:

```bash
pip install "rkoren-kitchen[dvc]"

kitchen init my-project \
  --source kaggle \
  --competition my-competition \
  --with-dvc
```

This scaffolds three extra files alongside the normal project structure:

```
my-project/
├── dvc.yaml          # pipeline stage definitions
├── .dvcignore        # tells DVC what not to track (like .gitignore)
└── .dvc/
    └── config        # S3 remote placeholder — edit this first
```

### Existing project

Use `kitchen dvc init` to add DVC to a project that was created without
`--with-dvc`. It only writes the DVC files — it does not touch your
`src/`, `menu.yaml`, or any other project code:

```bash
cd my-existing-project
pip install "rkoren-kitchen[dvc]"
kitchen dvc init
```

Pass `--kaggle` explicitly if your `menu.yaml` doesn't have
`data.source: kaggle` but you want the Kaggle-variant `dvc.yaml`:

```bash
kitchen dvc init --kaggle
```

---

## Remote setup

Before you can `dvc push` or `dvc pull`, point the remote at your S3 bucket.
The scaffold writes a placeholder — replace `YOUR-BUCKET`:

```bash
dvc remote modify s3remote url s3://my-project-data/dvc
```

This updates `.dvc/config`. Commit it:

```bash
git add .dvc/config
git commit -m "configure DVC S3 remote"
```

If your bucket is in a non-default region:

```bash
dvc remote modify s3remote region us-east-1
```

AWS credentials are read from the environment or `~/.aws/credentials` — the
same credentials used by `aws s3` and the `recipes` CLI.

---

## `dvc repro` vs `kitchen run`

Both are valid entry points. The difference is caching and dependency tracking.

### `kitchen run <stage>`

Runs one stage unconditionally, every time:

```bash
kitchen run train       # always re-trains, even if nothing changed
kitchen run evaluate    # always re-evaluates
```

Use this when you want to force a re-run regardless of input hashes — e.g.
iterating on `src/train/run.py` without waiting for DVC to detect the change,
or when you don't have DVC installed.

### `dvc repro`

Runs the full pipeline but skips stages whose inputs haven't changed:

```bash
dvc repro               # run all stages; skip unchanged ones
dvc repro train         # run from `train` stage onward
dvc repro evaluate      # run only `evaluate` (and its dependencies if stale)
```

A stage is re-run when any of its declared `deps` or `params` change:

```yaml
# dvc.yaml
stages:
  train:
    cmd: kitchen run train      # ← same command kitchen run uses
    deps:
      - src/train/run.py        # code change → re-run
      - data/processed/         # upstream data change → re-run
    params:
      - model                   # any key under `model:` in menu.yaml → re-run
    outs:
      - models/
```

Because each DVC stage's `cmd` is a `kitchen run` command, you can always
drop back to `kitchen run <stage>` when you want to force execution or debug
without the DVC layer.

---

## Push and pull

After running the pipeline, push processed data and models to S3:

```bash
dvc push               # upload data/processed/ and models/ to S3
```

A teammate clones the repo and restores the same data without re-running the
pipeline:

```bash
git clone https://github.com/you/my-project
cd my-project
pip install "rkoren-kitchen[dvc]"
dvc pull               # download data/processed/ and models/ from S3
```

Push after each significant run so CI and teammates always have a matching
data snapshot for the current commit.

---

## Kaggle vs non-Kaggle pipelines

The scaffolded `dvc.yaml` differs slightly by source type.

### Kaggle projects (`--source kaggle`)

Raw data is pinned by competition slug and re-downloaded on demand via
`kitchen ingest` — there is no DVC ingest stage:

```yaml
stages:
  # No ingest stage — kitchen ingest re-downloads from Kaggle API

  features:
    cmd: kitchen run features
    deps: [src/features/run.py, data/raw/]
    params: [features]
    outs: [data/processed/]

  train:
    cmd: kitchen run train
    deps: [src/train/run.py, data/processed/]
    params: [model]
    outs: [models/]

  evaluate:
    cmd: kitchen run evaluate
    deps: [src/evaluate/run.py, models/, data/processed/]
    params: [model]
    metrics:
      - metrics.json:
          cache: false

  submit:
    cmd: kitchen submit
    deps: [models/, data/raw/]
    outs: [submissions/]
```

`metrics.json` is declared with `cache: false` — DVC knows it is an output
(so `evaluate` depends on it for `submit`) but does not cache it in S3;
MLflow owns the metric history.

### Non-Kaggle projects (`--source local` or `--source s3`)

An ingest stage placeholder is included but commented out, because the data
source is project-specific:

```yaml
stages:
  # Uncomment and customise for script-driven ingest (custom API, S3 bucket, etc.).
  # ingest:
  #   cmd: python src/ingest/run.py
  #   deps:
  #     - src/ingest/run.py
  #   outs:
  #     - data/raw/

  features:
    ...
```

Uncomment and fill in the `ingest` stage if your data comes from a repeatable
source (an API, an S3 prefix, a database query). If you load data manually,
run `dvc add data/raw/` after placing your files — DVC tracks the directory
without needing a pipeline stage.

---

## External / API feature inputs (the "third data category")

`data/raw/` (immutable source) and `data/processed/` (regenerable from raw) don't
cover features pulled from a **live API** — KenPom ratings, an odds feed, weather.
Those aren't regenerable from raw (they need the API and are *point-in-time*), and
they aren't Kaggle raw. Caching them in `data/processed/` is the footgun behind a
real bug: that directory is gitignored and rebuilt, so CI's `dvc pull` never restores
the cache and the features stage **silently trains baseline-only**.

The pattern (`kitchen.ingest`):

```python
from kitchen.ingest import cached_fetch, require_external

# Fetch once, cache to a project-chosen, DVC-tracked path. Skip-if-present means the
# snapshot is pulled exactly once and CI reuses it after `dvc pull` — never re-fetched
# to a different "now". Encode the as-of key (season/date) in the filename.
ratings = cached_fetch(fetch_kenpom_2026, "data/kenpom/kenpom_2026.parquet")

# At the merge site, guard the input. A missing cache is a loud, explained error —
# never a silent skip that degrades the model to baseline-only.
require_external("data/kenpom/kenpom_2026.parquet")
```

Then track the directory once so CI restores it:

```bash
dvc add data/kenpom/      # `dvc pull` now restores it in CI, like data/raw/
```

The platform owns the cache/snapshot/restore semantics; **you** supply the fetch (it's
source-specific — a KenPom scrape isn't an odds API) and choose the path. There's no
`kitchen ingest` subcommand for this, precisely because the fetch is project code the
platform can't drive from config.

---

## CI integration

When using DVC in CI, runners need the versioned data before training.

**Kaggle projects** — skip `dvc pull`; data comes from `kitchen ingest`:

```yaml
# .github/workflows/train-evaluate.yml
- name: Download competition data
  run: kitchen ingest
  env:
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
```

**Non-Kaggle projects** — pull from S3 before training:

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
    aws-region: us-east-1

- name: Pull DVC data
  run: |
    pip install "rkoren-kitchen[dvc]"
    dvc pull
```

`kitchen init --with-dvc --ci` scaffolds the right variant automatically.

---

## Useful commands

```bash
# Run
dvc repro                    # full pipeline, skip unchanged stages
dvc repro train              # from `train` stage onward
dvc status                   # show which stages are stale

# Data
dvc push                     # upload outputs to S3
dvc pull                     # download outputs from S3
dvc add data/raw/            # manually track a directory (no stage needed)

# Inspect
dvc params diff              # show params that changed since last run
dvc metrics show             # show tracked metrics (metrics.json)
dvc dag                      # print the dependency graph

# Troubleshooting
dvc doctor                   # check DVC installation and config
dvc remote list              # list configured remotes
dvc gc -w                    # clean up old cached data (keep current workspace)
```

---

## Troubleshooting

**`dvc: command not found`**
```bash
pip install "rkoren-kitchen[dvc]"
```

**`ERROR: failed to push data to the cloud`** — check credentials and bucket name:
```bash
aws s3 ls s3://my-project-data/dvc/    # verify access
dvc remote list                         # verify URL matches
```

**Stage re-runs even though nothing changed** — DVC compares content hashes,
not timestamps. If `data/raw/` contents change between runs (e.g. a Kaggle
download adds a file), downstream stages are correctly invalidated.

**`dvc pull` after `git checkout` gives wrong data** — always run
`dvc checkout` after switching branches to restore the data that matches
the checked-out `dvc.lock`:
```bash
git checkout other-branch
dvc checkout          # restores data/processed/ and models/ for this branch
```
