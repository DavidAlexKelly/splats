#!/usr/bin/env python3
"""
Train again from a finished result's scene bundle, skipping frame extraction and the camera
solve (~50 min saved), or build only its scaffold.

    python retrain.py --video patio.mp4 --scene splats/patio_v3/scene --out NEW_FOLDER
    python retrain.py ... --scaffold-only            # Claude's scaffold only, no training
    python retrain.py ... --decision path/to/scaffold_decision.json

The training images are re-cut from the original video exactly as before (same frames,
same views). Training uses the CURRENT code and config.yaml (the bundle's config is only used
for how the images were cut), plus --overrides.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixer_refine  # noqa: E402
import splat_pipeline as sp  # noqa: E402
import train_gsplat as tg  # noqa: E402

log = logging.getLogger("retrain")
CUT_SECTIONS = ("extract", "select", "slice", "sfm")  # how the images/cameras were made


def build_config(scene: Path, overrides: dict, decision: str | None, work: Path) -> Path:
    app_cfg = Path(os.environ.get("SPLAT_CONFIG", HERE / "config.yaml"))
    cfg = sp.deep_merge(sp.DEFAULT_CONFIG, yaml.safe_load(app_cfg.read_text()) or {})
    old = yaml.safe_load((scene / "config.yaml").read_text()) or {}
    for k in CUT_SECTIONS:
        if k in old:
            cfg[k] = sp.deep_merge(cfg.get(k, {}), old[k])
    if os.environ.get("COLMAP_BIN"):
        cfg["sfm"]["colmap_bin"] = os.environ["COLMAP_BIN"]
    cfg = sp.deep_merge(cfg, overrides or {})
    if decision:  # absolute: the trainer runs in the job folder
        cfg = sp.deep_merge(cfg, {"train": {"custom": {
            "scaffold_decision": str(Path(decision).resolve())}}})
    work.mkdir(parents=True, exist_ok=True)
    path = work / "retrain_config.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                                   sort_keys=False))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True, help="a result's scene/ bundle")
    ap.add_argument("--out", type=Path, required=True, help="folder for the results")
    ap.add_argument("--work", type=Path, default=Path("work_retrain"))
    ap.add_argument("--overrides", default="", help="JSON merged over config.yaml")
    ap.add_argument("--decision", default="", help="scaffold_decision.json to reuse")
    ap.add_argument("--scaffold-only", action="store_true", dest="scaffold_only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(message)s")
    over = json.loads(args.overrides) if args.overrides.strip() else {}
    cfg_path = build_config(args.scene, over, args.decision or None, args.work)
    cfg = sp.load_config(cfg_path)
    args.out.mkdir(parents=True, exist_ok=True)

    data_dir = fixer_refine.rebuild_images(args.video, args.scene, args.work, cfg)
    if args.scaffold_only:
        s = tg.load_settings(str(cfg_path), {})
        s["_scene_name"] = args.video.stem
        info = tg.scaffold_only(data_dir, args.out, s)
        print("SCAFFOLD_ONLY_DONE " + json.dumps({k: info.get(k) for k in ("source", "rules")},
                                                 default=str))
        return

    # the bundle stands in for the stages before training
    job = sp.Job(args.video, args.work, args.work / "_out", cfg)
    assert job.dir == Path(data_dir).resolve(), (job.dir, data_dir)
    for name in ("sfm_stats.json", "intrinsics.json", "selection.json", "video.json"):
        if (args.scene / name).exists() and not (job.dir / name).exists():
            shutil.copy2(args.scene / name, job.dir / name)
    for stage in ("extract", "select", "slice", "sfm", "gate"):
        job.mark(stage, {"from_scene_bundle": str(args.scene)})
    import foundry_app
    foundry_app.setup_pipeline_logging()
    sp.log.propagate = False  # it prints itself; don't print every line twice
    stage_out = args.work / "_out"
    res = sp.run_pipeline([args.video], cfg, args.work, stage_out)[args.video.name]
    stem = args.video.stem
    for local, rel in foundry_app.result_files(stem, res, stage_out):
        target = args.out / Path(rel).relative_to(stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
    print("RETRAIN_DONE " + json.dumps({"status": res["status"], "error": res.get("error")}))
    sys.exit(0 if res["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
