# CI/CD Integration

`kitchen` scaffolds a GitHub Actions workflow that trains, evaluates, and reports on every push — with no custom hosting required.

## Scaffold the workflow

Pass `--ci` when initialising a project to generate `.github/workflows/train-evaluate.yml`:

```bash
kitchen init my-competition --source kaggle --competition spaceship-titanic --ci
```

The workflow runs on every push to `main` and on every pull request:

| Step | What it does |
|---|---|
| Ingest | Downloads competition data via `kitchen ingest` |
| Train | Runs `kitchen run train` and logs to MLflow |
| Evaluate | Runs `kitchen run evaluate` and writes `metrics.json` |
| Submit *(optional)* | Runs `kitchen submit --wait` when triggered via `workflow_dispatch` |
| Report | Runs `kitchen report` and appends to the GitHub Actions job summary |
| PR comment | Posts (or updates) a comment with the metrics table and delta vs. `main` |
| Artifacts | Uploads `metrics.json` and the HTML drift report as downloadable artifacts |

### Comparison baseline

`kitchen report` shows a delta column against a baseline. There are two ways to supply it:

- **`--compare <path>`** — a base `metrics.json`. The scaffolded CI uses this, downloading the previous successful `main` run's `metrics.json` as a GitHub artifact. This is the default because the scaffold's MLflow store is local SQLite (`sqlite:///mlruns.db`), which is **not** persisted across CI jobs — so a PR job has no champion in the registry to compare against.
- **`--compare champion`** (GH-011) — auto-fetches the registry champion's metrics. Use this when `MLFLOW_TRACKING_URI` points at a **remote** MLflow server (so the champion persists across runs), or locally/ad-hoc. If no champion is registered yet, it warns and falls back to a plain report (exit 0) rather than failing — so it's safe to wire into CI on a remote-tracking project.

## Secrets management

### Required secrets

| Secret | Where to get it |
|---|---|
| `KAGGLE_USERNAME` | Your Kaggle account username |
| `KAGGLE_KEY` | API token from [kaggle.com](https://www.kaggle.com/settings) → Account → API |

### Repository secrets vs. GitHub Environments

There are two places to store these secrets in GitHub:

| | Repository secrets | GitHub Environments |
|---|---|---|
| Scope | Available to every workflow in the repo | Scoped to a named environment (`staging`, `production`) |
| Approval gates | No | Yes — require a reviewer before the job runs |
| Branch restriction | No | Yes — restrict which branches can deploy to an environment |
| Audit trail | Basic | Full deployment log per environment |

**Recommendation: use GitHub Environments.** Repository secrets are available to any workflow branch, including forks on public repos. Environment secrets are scoped: the `staging` environment runs on PRs, the `production` environment runs on `main`. A leaked branch can never access `production` credentials.

### Setting up environments (recommended)

**Step 1 — Create the environments**

1. Go to **Settings → Environments**.
2. Click **New environment**, name it `staging`, and click **Configure environment**.
3. Leave the protection rules empty for `staging` (PRs run automatically).
4. Repeat, creating a `production` environment.
5. Under **Deployment branches and tags** for `production`, select **Selected branches** and add `main`.

**Step 2 — Add secrets to each environment**

For both `staging` and `production`:

1. Open the environment (Settings → Environments → select the environment).
2. Click **Add secret**.
3. Add `KAGGLE_USERNAME` (your Kaggle username).
4. Add `KAGGLE_KEY` (your Kaggle API token — download from kaggle.com → Account → API).

**Step 3 — Wire the CI workflow to an environment**

The scaffolded `.github/workflows/train-evaluate.yml` uses repository secrets by default. To switch to Environment secrets, add an `environment:` key to the job:

```yaml
jobs:
  train-evaluate:
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
    permissions:
      contents: read
      pull-requests: write
```

This expression picks `production` for pushes to `main` and `staging` for everything else (PRs, other branches). No other changes are needed — `secrets.KAGGLE_USERNAME` and `secrets.KAGGLE_KEY` resolve from the environment automatically.

!!! tip "Optional: require approval for production"
    In **Settings → Environments → production**, enable **Required reviewers** and add yourself. Every push to `main` will then pause and wait for your approval before the ingest and submit steps run — useful if you want to gate Kaggle submissions manually.

## Branch protection

Gate PR merges on the evaluation job passing so that no change lands without a recorded metric.

1. Go to **Settings → Branches → Add branch ruleset** (or the legacy **Branch protection rules**).
2. Set the branch name pattern to `main`.
3. Enable **Require status checks to pass before merging**.
4. In the search box, type `train-evaluate` and select it. This is the job name inside the `Train and Evaluate — <project>` workflow.
5. Optionally enable **Require branches to be up to date before merging** so that the check always runs against the latest `main`.
6. Save the rule.

!!! tip "Finding the check name"
    The status check only appears in the search box after the workflow has run at least once on a pull request. Open a draft PR, let the workflow complete, then return to this settings page.

## Metric thresholds

You can fail the CI step when a metric drops below a threshold. Add a `thresholds` block to `params.yaml`:

```yaml
thresholds:
  val_accuracy: 0.80          # fail if below 0.80
  val_logloss:
    max: 0.50                 # fail if above 0.50
```

The `Report` step exits non-zero when any threshold is violated, which fails the `train-evaluate` check and blocks the PR merge.

## Manual Kaggle submission

The submit step is gated behind a `workflow_dispatch` input so it only runs when you deliberately trigger it:

1. Go to **Actions → Train and Evaluate — \<project\>**.
2. Click **Run workflow**.
3. Check **Submit to Kaggle leaderboard after evaluate**.
4. Run.

The leaderboard score is written to `metrics.json` and appears in the PR comment when `kitchen report` next runs.
