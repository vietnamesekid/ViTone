"""ViTone command line.

vitone data ingest [-c CONFIG] [-s SOURCE ...] [--work-dir DIR] [--workers N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from vitone.data.config import Config
from vitone.data.pipeline import run_ingest

DATA_STAGES = {"ingest": run_ingest}
log = logging.getLogger("vitone")


def _data(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    names = args.source or list(cfg.sources)
    if unknown := sorted(set(names) - cfg.sources.keys()):
        raise SystemExit(f"unknown source(s) {unknown}; defined in {args.config}: {sorted(cfg.sources)}")

    failed = []
    for name in names:  # one failing source does not stop the others
        out = args.work_dir / name
        try:
            rows = DATA_STAGES[args.stage](cfg.sources[name], out, args.workers)
        except Exception:
            log.exception("%s %s failed", args.stage, name)
            failed.append(name)
            continue
        hours = sum(r["duration"] for r in rows) / 3600
        print(f"{args.stage} {name}: {len(rows)} recordings, {hours:.2f} h -> {out / 'manifests'}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vitone")
    commands = ap.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="build training data from raw media")
    data.add_argument("stage", choices=DATA_STAGES)
    data.add_argument("-c", "--config", default="configs/data.yaml", help="dataset definition")
    data.add_argument("-s", "--source", action="append", help="source name (repeatable; default: all)")
    data.add_argument(
        "--work-dir", type=Path, default=Path("corpus/processed"), help="outputs go to WORK_DIR/<source>/"
    )
    data.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    data.set_defaults(func=_data)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
