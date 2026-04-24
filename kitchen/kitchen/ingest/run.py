"""DVC stage entrypoint: ingest raw data.

Run via:  python -m kitchen.ingest.run
or:       dvc repro ingest
"""
from pathlib import Path

import yaml

from kitchen.ingest import source_from_params
from kitchen.store import DataStore

PARAMS_PATH = Path("params.yaml")


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())["data"]
    store = DataStore()
    source = source_from_params(params)
    files = source.download(store.raw_dir)
    print(f"Ingested {len(files)} file(s) → {store.raw_dir}")


if __name__ == "__main__":
    main()
