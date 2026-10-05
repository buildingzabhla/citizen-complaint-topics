"""Shared helpers: configuration, paths, logging, seeds and caching."""
from __future__ import annotations

import logging
import pickle
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

log = logging.getLogger("pipeline")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    # gensim is very talkative at INFO level
    logging.getLogger("gensim").setLevel(logging.WARNING)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def path(rel: str) -> Path:
    """Resolve a path from the config relative to the repository root."""
    return ROOT / rel


def ensure_dirs() -> None:
    for d in (INTERIM, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)


def save(obj, name: str) -> Path:
    """Cache an intermediate result in data/interim/ so later steps can reuse it."""
    p = INTERIM / name
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load(name: str):
    p = INTERIM / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing - run the earlier pipeline steps first.")
    with open(p, "rb") as f:
        return pickle.load(f)


def exists(name: str) -> bool:
    return (INTERIM / name).exists()


@contextmanager
def timed(label: str):
    start = time.time()
    log.info("%s ...", label)
    yield
    log.info("%s finished in %.1f s", label, time.time() - start)
