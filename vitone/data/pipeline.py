"""Stage drivers: run a stage over every file of a source and write its manifests.

Layout per source (`out` = <work_dir>/<source>):
    out/audio/                            decoded FLAC for sources that need it
    out/manifests/
        recordings.jsonl.gz               Lhotse RecordingSet
        ingest.jsonl                      ingest measurements, one line per recording
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lhotse import RecordingSet
from tqdm import tqdm

from vitone.data.config import SourceConfig
from vitone.data.ingest import NotAudioError, ingest, recording_id, to_recording

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_ingest(src: SourceConfig, out: Path, workers: int = 1) -> list[dict]:
    """Ingest every file of a source. Manifests store absolute paths so they stay
    valid from any working directory. Files that fail are logged and reported after
    the manifests for the others are written. Files without audio (cover art, notes)
    are skipped with a warning."""
    out = out.resolve()
    manifest = out / "manifests" / "ingest.jsonl"
    cached = {r["recording_id"]: r for r in read_jsonl(manifest)}

    files = src.files()
    if not files:
        raise FileNotFoundError(f"no files match {src.path / src.glob}")
    ids = {path: recording_id(src.name, path.relative_to(src.root)) for path in files}
    clashes = sorted(rid for rid, n in Counter(ids.values()).items() if n > 1)
    if clashes:
        raise ValueError(f"different files map to the same recording id: {clashes}; rename them")

    results, failed = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ingest, path, rid, out / "audio", cached.get(rid)): path for path, rid in ids.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"ingest {src.name}"):
            try:
                results.append(fut.result().to_dict())
            except NotAudioError as e:
                log.warning("skipped: %s", e)
            except Exception:
                log.exception("ingest failed: %s", futures[fut])
                failed.append(str(futures[fut]))

    results.sort(key=lambda r: r["recording_id"])
    write_jsonl(manifest, results)
    RecordingSet.from_recordings(to_recording(r) for r in results).to_file(out / "manifests" / "recordings.jsonl.gz")
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed to ingest: {failed}")
    return results
