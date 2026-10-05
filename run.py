"""Run the complaint topic analysis.

Usage:
    python run.py                       # all steps
    python run.py --steps data          # a single step
    python run.py --steps topics,report # several steps (earlier results are reused)

Every step stores its output in data/interim/ (intermediate files) or results/
(figures and tables), so a later step can be repeated without redoing the earlier ones.
"""
from __future__ import annotations

import argparse
import importlib
import time

from pipeline.utils import ensure_dirs, load_config, log, set_seeds, setup_logging

STEPS = ["data", "preprocess", "vectorize", "topics", "evaluate", "report"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default="all",
                        help="comma-separated steps out of: " + ", ".join(STEPS))
    parser.add_argument("--config", default=None, help="path to the config file")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    ensure_dirs()
    steps = STEPS if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"unknown step(s): {unknown}")

    start = time.time()
    for step in steps:
        set_seeds(cfg["seed"])
        log.info("=" * 20 + f" step: {step} " + "=" * 20)
        importlib.import_module(f"pipeline.{step}").run(cfg)
    log.info("done in %.1f min", (time.time() - start) / 60)


if __name__ == "__main__":   # required: spaCy and gensim start worker processes
    main()
