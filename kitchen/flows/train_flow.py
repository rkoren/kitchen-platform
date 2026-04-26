import yaml
from prefect import flow, task

from kitchen.ingest import source_from_params
from kitchen.store import DataStore
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
        raise RuntimeError("Project must implement src/features/run.py with a build(params, store) function") from e
    build(params, store)


@task(name="train")
def train_model(params: dict, store: DataStore, tracker: Tracker) -> object:
    try:
        from src.train.run import train  # project-defined
    except ImportError as e:
        raise RuntimeError("Project must implement src/train/run.py with a train(params, store, tracker) function") from e
    return train(params, store, tracker)


@task(name="evaluate")
def evaluate_model(model: object, params: dict, store: DataStore) -> dict:
    try:
        from src.evaluate.run import evaluate  # project-defined
    except ImportError as e:
        raise RuntimeError("Project must implement src/evaluate/run.py with an evaluate(model, params, store) function") from e
    return evaluate(model, params, store)


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
