"""Prefect flow: end-to-end training pipeline."""
from prefect import flow, task


@task
def ingest():
    raise NotImplementedError


@task
def build_features():
    raise NotImplementedError


@task
def train_model():
    raise NotImplementedError


@task
def evaluate_model():
    raise NotImplementedError


@flow(name="train")
def train_pipeline():
    raw = ingest()
    processed = build_features(raw)
    model = train_model(processed)
    evaluate_model(model, processed)


if __name__ == "__main__":
    train_pipeline()
