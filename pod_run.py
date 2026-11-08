#!/usr/bin/env python3
"""
One command for a rented GPU (RunPod, Vast.ai, or any Linux box with an NVIDIA GPU):
360 videos in, splats out, then the optional Difix AI fill-in.

    python /opt/app/pod_run.py                     # every video in /workspace/videos
    python /opt/app/pod_run.py patio.mp4           # just these videos
    python /opt/app/pod_run.py --no-difix          # real-data splats only
    python /opt/app/pod_run.py --difix-only patio_v2   # refine an existing result again
    python /opt/app/pod_run.py --selftest          # check the GPU, COLMAP, gsplat and Difix first
    python /opt/app/pod_run.py --stop-when-done    # stop the pod at the end (stops the billing)

Folders (override with --videos/--out/--work or SPLAT_VIDEOS/SPLAT_RESULTS/SPLAT_WORK):
    /workspace/videos   put your .mp4 files here
    /workspace/splats   results: <name>/<name>.ply (+ scene/, eval/, stats, log) and
                        <name>_difix/<name>_difix.ply (+ difix_preview/, difix_stats.json)
    /workspace/work     temporary working files (tens of GB per 8K video; cleaned up)

Behaves like the Foundry transforms: nothing is ever overwritten (a re-run becomes patio_v2,
patio_v3, ...), and a video already processed with the same code + settings is skipped unless
you pass --rerun or change --overrides. Difix is licensed for non-commercial use only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent
PY = sys.executable
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}  # exported equirectangular 360 video
RUN_KEY = "run_key.json"


def say(msg: str):
    print(f"[pod_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- helpers
def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(APP.glob("*.py")) + sorted(APP.glob("*.yaml")):
        if f.name == "pod_run.py":  # changes to this runner don't change the results
            continue
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def run_key(*parts) -> str:
    h = hashlib.sha256(code_hash().encode())
    for p in parts:
        h.update(json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:16]


def stem_of(folder: str) -> str:
    return re.sub(r"_v\d+$", "", folder)


def next_version(name: str, results: Path) -> str:
    existing = {p.name for p in results.iterdir() if p.is_dir()} if results.exists() else set()
    if name not in existing:
        return name
    pat = re.compile(rf"^{re.escape(name)}_v(\d+)$")
    taken = [int(m.group(1)) for x in existing if (m := pat.match(x))]
    return f"{name}_v{max(taken + [1]) + 1}"


def published(results: Path) -> list[dict]:
    out = []
    for f in sorted(results.glob(f"*/{RUN_KEY}")) if results.exists() else []:
        try:
            out.append({**json.loads(f.read_text()), "folder": f.parent.name})
        except (OSError, ValueError):
            pass
    return out


def child_env(work: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONUNBUFFERED": "1",
        "SPLAT_WORK": str(work),
        "SPLAT_KEEP_WORK": "0",
        "COLMAP_BIN": env.get("COLMAP_BIN", "colmap"),
        "HF_HOME": env.get("HF_HOME", "/opt/hf"),
        "HF_HUB_OFFLINE": "1",
        "TORCH_HOME": env.get("TORCH_HOME", "/opt/torchhub"),
        "HF_MODULES_CACHE": str(work / "hf_modules"),  # Difix's model code is copied here
    })
    return env


def stream(cmd: list, env: dict, log_file: Path) -> int:
    """Run a command, showing its output and saving it to log_file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    say("running: " + " ".join(map(str, cmd)))
    with open(log_file, "a", encoding="utf-8") as lf:
        proc = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        return proc.wait()


def disk_check(path: Path, need_gb: float = 40):
    free = shutil.disk_usage(path).free / 1e9
    if free < need_gb:
        say(f"WARNING: only {free:.0f} GB free in {path}; an 8K video can need 30-60 GB of "
            "working space. Give the pod a bigger volume if a run fails with 'No space left'.")


# --------------------------------------------------------------------------- stages
def make_splat(video: Path, a, results: Path, work: Path, key: str) -> tuple[str, bool]:
    """Main pipeline for one video. Returns (published folder name, ok)."""
    stem = video.stem
    staging = Path(tempfile.mkdtemp(prefix="pod_", dir=work))
    inp, out = staging / "in", staging / "out"
    inp.mkdir()
    out.mkdir()
    (inp / video.name).symlink_to(video.resolve())
    env = child_env(work)
    env.update({"SPLAT_MODE": "local", "SPLAT_LOCAL_INPUT": str(inp),
                "SPLAT_LOCAL_OUTPUT": str(out), "SPLAT_FORCE": "1",
                "SPLAT_CONFIG_OVERRIDES": json.dumps(a.overrides)})
    log = results / "_logs" / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    code = stream([PY, APP / "foundry_app.py"], env, log)
    ok = (out / stem / f"{stem}.ply").exists()
    name = next_version(stem, results)
    dest = results / name
    if (out / stem).exists():
        shutil.copytree(out / stem, dest)
    dest.mkdir(parents=True, exist_ok=True)
    if name != stem:  # patio.ply -> patio_v2.ply, patio.json -> patio_v2.json
        for f in list(dest.iterdir()):
            if f.is_file() and f.stem == stem:
                f.rename(dest / (name + f.suffix))
    if not ok:
        (dest / "FAILED.txt").write_text(
            f"No splat was produced for {video.name} (exit code {code}). Full log: {log}\n")
    shutil.copy2(log, dest / "run.log")
    (dest / RUN_KEY).write_text(json.dumps(
        {"video": video.name, "run_key": key, "overrides": a.overrides, "rerun": a.rerun,
         "status": "ok" if ok else "failed", "kind": "splat",
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    shutil.rmtree(staging, ignore_errors=True)
    say(f"{video.name}: {'OK' if ok else 'FAILED'} -> {dest}")
    return name, ok


def refine(folder: str, a, results: Path, videos: Path, work: Path) -> tuple[str, bool]:
    """Difix refinement of an existing result folder."""
    src = results / folder
    ply, scene = src / f"{folder}.ply", src / "scene"
    if not ply.exists() or not (scene / "config.yaml").exists():
        raise SystemExit(f"{src} has no {ply.name} or scene/ folder (results made before the "
                         "scene-bundle update can't be refined: run the video again)")
    info = json.loads((src / RUN_KEY).read_text()) if (src / RUN_KEY).exists() else {}
    vname = info.get("video") or ""
    video = videos / vname if vname else None
    if not video or not video.exists():
        hits = [p for p in videos.iterdir() if p.stem == stem_of(folder)
                and p.suffix.lower() in VIDEO_EXTS]
        if len(hits) != 1:
            raise SystemExit(f"Can't find the video for {folder} in {videos} (Difix re-cuts the "
                             "training images from it)")
        video = hits[0]
    key = run_key("difix", folder, a.difix_overrides)
    done = [p for p in published(results) if p.get("kind") == "difix"
            and p.get("run_key") == key and p.get("status") == "ok"]
    if done and not a.rerun:
        say(f"{folder}: already refined with these settings ({done[0]['folder']}); skipping")
        return done[0]["folder"], True
    name = next_version(f"{folder}_difix", results)
    dest = results / name
    dest.mkdir(parents=True)
    dwork = work / f"difix_{name}"
    log = results / "_logs" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    code = stream([PY, APP / "difix_refine.py", "--video", video, "--scene", scene,
                   "--ply", ply, "--out", dest, "--work", dwork,
                   "--overrides", json.dumps(a.difix_overrides)], child_env(work), log)
    ok = code == 0 and (dest / "splat.ply").exists()
    if ok:
        (dest / "splat.ply").rename(dest / f"{name}.ply")
    else:
        (dest / "FAILED.txt").write_text(f"Difix refinement of {folder} failed (exit code "
                                         f"{code}). Full log: {log}\n")
    shutil.copy2(log, dest / "run.log")
    shutil.rmtree(dwork, ignore_errors=True)
    (dest / RUN_KEY).write_text(json.dumps(
        {"source": folder, "video": video.name, "run_key": key, "kind": "difix",
         "overrides": a.difix_overrides, "rerun": a.rerun, "status": "ok" if ok else "failed",
         "note": "Contains AI-generated content (NVIDIA Difix3D+, non-commercial licence).",
         "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    say(f"{folder}: Difix {'OK' if ok else 'FAILED'} -> {dest}")
    return name, ok


def stop_pod():
    """Stop this machine so it stops billing (RunPod or Vast.ai). Storage is kept."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod:
        if shutil.which("runpodctl"):
            if subprocess.run(["runpodctl", "stop", "pod", pod]).returncode == 0:
                return True
        key = os.environ.get("RUNPOD_API_KEY")
        if key:
            q = json.dumps({"query": 'mutation { podStop(input: {podId: "%s"}) '
                                     '{ id desiredStatus } }' % pod}).encode()
            req = urllib.request.Request("https://api.runpod.io/graphql", data=q, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            try:
                urllib.request.urlopen(req, timeout=30).read()
                return True
            except Exception as e:  # noqa: BLE001
                say(f"RunPod API stop failed: {e}")
    cid, ckey = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")
    if cid and ckey and shutil.which("vastai"):
        return subprocess.run(["vastai", "stop", "instance", cid, "--api-key", ckey]).returncode == 0
    return False


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("videos_to_run", nargs="*", help="video file names (default: all)")
    ap.add_argument("--videos", type=Path, default=Path(os.environ.get("SPLAT_VIDEOS", "/workspace/videos")))
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SPLAT_RESULTS", "/workspace/splats")))
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("SPLAT_WORK", "/workspace/work")))
    ap.add_argument("--no-difix", action="store_true", help="skip the Difix AI fill-in")
    ap.add_argument("--difix-only", nargs="+", metavar="FOLDER",
                    help="only refine these existing result folders with Difix")
    ap.add_argument("--overrides", type=json.loads, default={},
                    help='config.yaml overrides as JSON, e.g. \'{"train":{"custom":{"cap_max":5000000}}}\'')
    ap.add_argument("--difix-overrides", type=json.loads, default={},
                    help='Difix settings as JSON, e.g. \'{"difix":{"rounds":2}}\'')
    ap.add_argument("--rerun", action="store_true", help="process again even if already done")
    ap.add_argument("--selftest", action="store_true", help="run the self-test first")
    ap.add_argument("--stop-when-done", action="store_true",
                    help="stop the pod afterwards (RunPod / Vast.ai), even if something failed")
    a = ap.parse_args()
    for d in (a.videos, a.out, a.work):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: list[str] = []
    failed = False
    try:
        if a.selftest:
            env = child_env(a.work)
            code = stream([PY, APP / "selftest.py", "--work", a.work / "selftest"]
                          + ([] if a.no_difix else ["--with-difix"]), env,
                          a.out / "_logs" / f"selftest_{time.strftime('%Y%m%d_%H%M%S')}.log")
            shutil.rmtree(a.work / "selftest", ignore_errors=True)
            if code != 0:
                raise SystemExit("Self-test failed (see above); nothing was processed.")
        if a.difix_only:
            for folder in a.difix_only:
                name, ok = refine(folder, a, a.out, a.videos, a.work)
                summary.append(f"{'OK    ' if ok else 'FAILED'} {name}")
                failed |= not ok
        else:
            failed |= process_videos(a, summary)
    finally:
        mins = (time.time() - t0) / 60
        say(f"finished in {mins:.0f} min")
        for line in summary:
            say("  " + line)
        if a.stop_when_done:
            say("stopping the pod (your files on the volume are kept)...")
            if not stop_pod():
                say("couldn't stop the pod automatically: stop it in the RunPod / Vast.ai "
                    "website, or it keeps billing")
    sys.exit(1 if failed else 0)


def process_videos(a, summary: list[str]) -> bool:
    """Main pipeline (+ Difix) for each video; returns True if anything failed."""
    failed = False
    videos = sorted(p for p in a.videos.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if a.videos_to_run:
        want = set(a.videos_to_run)
        videos = [v for v in videos if v.name in want or v.stem in want]
        missing = want - {v.name for v in videos} - {v.stem for v in videos}
        if missing:
            raise SystemExit(f"Not found in {a.videos}: {sorted(missing)}")
    if not videos:
        raise SystemExit(f"No videos in {a.videos}. Upload some first (see RENTING_A_GPU.md).")
    say(f"{len(videos)} video(s): {[v.name for v in videos]}; results -> {a.out}")
    disk_check(a.work)
    prev = published(a.out)
    for v in videos:
        key = run_key("splat", v.name, a.overrides)
        same = [p for p in prev if p.get("kind", "splat") == "splat" and p.get("video") == v.name
                and p.get("run_key") == key and p.get("status") == "ok"]
        if same and not a.rerun:
            folder = same[-1]["folder"]
            say(f"{v.name}: already done with this code + settings ({folder}); "
                "skipping (--rerun to redo)")
        else:
            folder, ok = make_splat(v, a, a.out, a.work, key)
            summary.append(f"{'OK    ' if ok else 'FAILED'} {folder}")
            failed |= not ok
            if not ok:
                continue
        if not a.no_difix:
            name, ok = refine(folder, a, a.out, a.videos, a.work)
            summary.append(f"{'OK    ' if ok else 'FAILED'} {name}")
            failed |= not ok
    return failed


if __name__ == "__main__":
    main()
