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
| Artifacts | Uploads `metrics.json` and the Evidently HTML report as downloadable artifacts |

## Required secrets

Add these to your repository (or a GitHub Environment):

| Secret | Where to get it |
|---|---|
| `KAGGLE_USERNAME` | Your Kaggle account username |
| `KAGGLE_KEY` | API token from kaggle.com → Account → API |

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
