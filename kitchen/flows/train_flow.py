import yaml
from prefect import flow, task

from kitchen.ingest import source_from_params
from kitchen.store import DataStore
from kitchen.submit import log_submission as _log_submission
from kitchen.tracking import Tracker


@task(name="ingest")
def ingest(params: dict, store: DataStore) -> list[str]:
    source = source_from_params(params["data"])
    return source.download(store.raw_dir)


@task(name="features")
def build_features(params: dict, store: DataStore) -> None:
    try:
        from src.features.run import build  # project-defined
    except ImportError as e:
        raise RuntimeError(
            "Project must implement src/features/run.py with a build(params, store) function"
        ) from e
    build(params, store)


@task(name="train")
def train_model(params: dict, store: DataStore, tracker: Tracker) -> object:
    try:
        from src.train.run import train  # project-defined
    except ImportError as e:
        raise RuntimeError(
            "Project must implement src/train/run.py with a train(params, store, tracker) function"
        ) from e
    return train(params, store, tracker)


@task(name="evaluate")
def evaluate_model(model: object, params: dict, store: DataStore) -> dict:
    try:
        from src.evaluate.run import evaluate  # project-defined
    except ImportError as e:
        raise RuntimeError(
            "Project must implement src/evaluate/run.py with an evaluate(model, params, store) function"
        ) from e
    return evaluate(model, params, store)


@task(name="submit")
def submit_submission(model: object, params: dict, store: DataStore) -> dict[str, float]:
    """Generate a submission, validate it, log as an MLflow artifact, and optionally upload.

    Expects the project to implement ``src/submit/run.py`` with::

        def generate(model, params, store) -> tuple[pd.DataFrame, Path]:
            ...  # build submission DataFrame, write to disk, return (df, path)

    The ``submission`` section of params.yaml configures upload behaviour::

        submission:
          sample_file: SampleSubmissionStage1.csv   # relative to data/raw/
          id_col: ID
          target_col: Pred
          competition: march-machine-learning-mania-2026  # omit to skip upload
          message: "my run description"
          fetch_lb_score: false   # true to poll Kaggle after uploading
    """
    import pandas as pd

    try:
        from src.submit.run import generate  # project-defined
    except ImportError as e:
        raise RuntimeError(
            "Project must implement src/submit/run.py with a"
            " generate(model, params, store) -> (df, path) function"
        ) from e

    sub_df, sub_path = generate(model, params, store)
    sub_cfg = params.get("submission", {})
    sample_file = sub_cfg.get("sample_file", "SampleSubmissionStage1.csv")
    sample = pd.read_csv(store.raw_dir / sample_file)
    return _log_submission(
        submission=sub_df,
        sample=sample,
        file_path=sub_path,
        id_col=sub_cfg.get("id_col", "ID"),
        target_col=sub_cfg.get("target_col", "Pred"),
        competition=sub_cfg.get("competition"),
        message=sub_cfg.get("message", ""),
        fetch_lb_score=sub_cfg.get("fetch_lb_score", False),
    )


@flow(name="train")
def train_pipeline(params_file: str = "params.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    store = DataStore()
    tracker = Tracker(
        experiment=params.get("experiment", "default"),
        tracking_uri=params.get("mlflow", {}).get("tracking_uri"),
    )

    ingest(params, store)
    build_features(params, store)
    model = train_model(params, store, tracker)
    evaluate_model(model, params, store)


if __name__ == "__main__":
    train_pipeline()
