#!/usr/bin/env python3
"""
splat_stream.py - move result files through a Foundry stream as text records, and rebuild
them afterwards.

Every file is split into chunks; each chunk becomes one stream record:

    record_type   "chunk" (file data) or "run" (one summary record per video run)
    run_id        unique id of the pipeline run
    video         source video file name, e.g. "patio.mp4"
    file_path     path of the rebuilt file, e.g. "patio/patio.ply"
    chunk_index   0 .. chunk_count-1
    chunk_count   number of chunks for this file
    total_bytes   size of the complete file
    sha256        checksum of the complete file
    data_b64      base64 of this chunk's bytes (empty for "run" records)
    info          JSON text (run summary on "run" records, else "")
    created_at    ISO-8601 UTC time the record was written

Records can arrive in any order and may be duplicated by retries: reconstruction sorts by
chunk_index, drops duplicates, only rebuilds files whose chunks are all present, picks the
newest complete run per file, and verifies the checksum.

Local use (rebuild from an exported copy of the stream's records):
    python splat_stream.py rebuild records.parquet out_folder
    python splat_stream.py rebuild records.csv out_folder
    python splat_stream.py rebuild records.jsonl out_folder
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

FIELDS = ["record_type", "run_id", "video", "file_path", "chunk_index", "chunk_count",
          "total_bytes", "sha256", "data_b64", "info", "created_at"]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_records(local: Path, file_path: str, run_id: str, video: str,
                 chunk_bytes: int) -> Iterator[dict]:
    """Yield chunk records for one file without loading it all into memory."""
    size = local.stat().st_size
    count = max(1, -(-size // chunk_bytes))
    digest = sha256_file(local)
    with open(local, "rb") as f:
        for i in range(count):
            data = f.read(chunk_bytes)
            yield {"record_type": "chunk", "run_id": run_id, "video": video,
                   "file_path": file_path, "chunk_index": i, "chunk_count": count,
                   "total_bytes": size, "sha256": digest,
                   "data_b64": base64.b64encode(data).decode("ascii"), "info": "",
                   "created_at": now_iso()}


def run_record(run_id: str, video: str, summary: dict) -> dict:
    return {"record_type": "run", "run_id": run_id, "video": video, "file_path": "",
            "chunk_index": 0, "chunk_count": 0, "total_bytes": 0, "sha256": "",
            "data_b64": "", "info": json.dumps(summary), "created_at": now_iso()}


# --------------------------------------------------------------------------- #
# Reconstruction (pure Python; the Foundry transform uses the same rules in Spark)
# --------------------------------------------------------------------------- #
def plan(records: Iterable[dict]) -> dict:
    """Group chunk records -> {file_path: chosen run's {meta, chunks{index: b64}}}.
    Picks, per file_path, the newest run whose chunks are complete."""
    runs: dict[tuple, dict] = {}
    for r in records:
        if r.get("record_type") != "chunk":
            continue
        key = (r["file_path"], r["run_id"])
        g = runs.setdefault(key, {"file_path": r["file_path"], "run_id": r["run_id"],
                                  "video": r.get("video", ""),
                                  "chunk_count": int(r["chunk_count"]),
                                  "total_bytes": int(r["total_bytes"]), "sha256": r["sha256"],
                                  "latest": "", "chunks": {}})
        g["chunks"].setdefault(int(r["chunk_index"]), r["data_b64"])  # duplicates ignored
        g["latest"] = max(g["latest"], str(r.get("created_at") or ""))
    chosen: dict[str, dict] = {}
    for g in runs.values():
        if len(g["chunks"]) != g["chunk_count"]:
            continue  # incomplete (still streaming, or lost records)
        cur = chosen.get(g["file_path"])
        if cur is None or g["latest"] > cur["latest"]:
            chosen[g["file_path"]] = g
    return chosen


def rebuild(records: Iterable[dict], out_dir: Path) -> list[dict]:
    """Write every complete file under out_dir; returns a manifest."""
    manifest = []
    for path, g in sorted(plan(records).items()):
        target = out_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(target, "wb") as f:
            for i in range(g["chunk_count"]):
                data = base64.b64decode(g["chunks"][i])
                h.update(data)
                f.write(data)
        ok = h.hexdigest() == g["sha256"] and target.stat().st_size == g["total_bytes"]
        if not ok:
            target.unlink()
        manifest.append({"file_path": path, "run_id": g["run_id"], "video": g["video"],
                         "bytes": g["total_bytes"], "verified": ok})
    return manifest


def load_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data.get("values", data.get("data", []))
    import pandas as pd  # parquet/csv exports
    df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path, dtype=str)
    return df.fillna("").to_dict("records")


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "rebuild":
        print(__doc__)
        sys.exit(2)
    src, out = Path(sys.argv[2]), Path(sys.argv[3])
    files = sorted(src.glob("*")) if src.is_dir() else [src]
    records = [r for f in files if f.is_file() for r in load_records(f)]
    manifest = rebuild(records, out)
    for m in manifest:
        print(f"{'OK ' if m['verified'] else 'BAD'} {m['file_path']} ({m['bytes'] / 1e6:.1f} MB)")
    if not manifest:
        print("No complete files found in the records.")


if __name__ == "__main__":
    main()
