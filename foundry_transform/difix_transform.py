"""
Foundry lightweight transform: AI fill-in for splats the main transform (splat_transform.py)
has already produced, using NVIDIA Difix3D+. Runs in the same Docker image on the same GPU.

    videos dataset  ──►  splat_transform  ──►  splats dataset  ──►  difix_transform  ──►  difix dataset
                                               (real-only splats)                         (AI-filled splats)

Why a separate transform
  * The real-only splat stays untouched; the Difix version is published next to it, so you can
    compare them in the viewer and keep whichever is better per scene.
  * Re-run Difix with other settings on any existing result without redoing SfM + training.
  * The main pipeline stays free of the non-commercial Difix licence.

LICENCE: Difix code and weights are under the NVIDIA License: non-commercial research and
evaluation use only. Results contain AI-generated content (marked in difix_stats.json).

What it needs from each result: <folder>/<folder>.ply plus <folder>/scene/ (the scene bundle,
written by pipeline versions from the cube-map/rig update onwards; re-run older results once in
splat_transform with RERUN bumped) and the original video in the videos dataset.

Setup
  1. Create an empty output dataset (e.g. "splat_files_difix") and put its RID in DIFIX_DATASET.
  2. Put this file next to splat_transform.py, commit, press Build on the output dataset.

How it behaves
  * TARGETS = ["latest"]: the newest version of every video that has a scene bundle
    (patio, patio_v2, patio_v3 -> patio_v3). Or list folders by name: ["patio_v3", "garden"].
  * Output: <folder>_difix/<folder>_difix.ply, difix_stats.json (PSNR before/after on real
    photos), difix_preview/*.jpg (render | Difix | nearest real photo), run_key.json.
  * Skips targets already refined successfully with the same image + settings (run key, as
    in splat_transform); failed ones are tried again on the next build. Change DIFIX_OVERRIDES or bump RERUN to refine again; a re-run is
    published as <folder>_difix_v2, _v3, ... never overwriting.
"""
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from transforms.api import Input, Output, incremental, transform

# ----------------------------------------------------------------------------- CONFIG
VIDEOS_DATASET = "ri.foundry.main.dataset.9edeb675-87e7-4c9c-bac4-d6ba450db4f7"   # raw .mp4 files
SPLATS_DATASET = "ri.foundry.main.dataset.f53697ca-5c35-40ab-b1ae-fec6946b3ac0"   # splat_transform output
DIFIX_DATASET = "ri.foundry.main.dataset.REPLACE-WITH-YOUR-NEW-EMPTY-DATASET"      # output of this transform
IMAGE_NAME = "hello-world"     # same image as splat_transform (built with WITH_DIFIX=1, the default)
IMAGE_TAG = "0.0.1"
CPU_CORES = 8
MEMORY_GB = 32
GPU_TYPE = "NVIDIA_T4"
# "latest" = newest version of every video; or folder names from the splats dataset.
TARGETS = ["latest"]
# Overrides for the `difix:` settings (see DIFIX_DEFAULTS in difix_refine.py), e.g. a quick
# first look: {"difix": {"rounds": 2, "views_per_round": 120}}
DIFIX_OVERRIDES: dict = {}
# Check that Difix loads and runs on the GPU (~2 min) before refining anything.
SELFTEST = True
# Bump to refine the same targets again with unchanged settings.
RERUN = 0
# ------------------------------------------------------------------------------------

IMAGE_PYTHON = "/opt/venv/bin/python"
DIFIX_PYTHON = "/opt/difix/bin/python"
APP_DIR = Path("/opt/app")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
RUN_KEY_FILE = "run_key.json"

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- helpers
def _env(work: Path, home: Path) -> dict:
    """The image's own Python environment, without the transform runtime's variables and
    libraries (same rules as splat_transform)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "CONDA", "VIRTUAL_ENV"))}
    gpu_dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if d and ("nvidia" in d or "cuda" in d)]
    env.update({
        "LD_LIBRARY_PATH": ":".join(dict.fromkeys(gpu_dirs + [
            "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"])),
        "PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "VIRTUAL_ENV": "/opt/venv",
        "PYTHONUNBUFFERED": "1",
        "SPLAT_WORK": str(work),
        "COLMAP_BIN": "colmap",
        "HF_HOME": "/opt/hf",
        "HF_HUB_OFFLINE": "1",
        "TORCH_HOME": "/opt/torchhub",
        "HOME": str(home),
        "TORCH_EXTENSIONS_DIR": str(home / "torch_extensions"),
        "DIFIX_SRC": "/opt/difix/src",
        # Difix's model ships its own code, which diffusers copies into a modules cache
        # before importing it. /opt/hf is read-only at run time, so use a writable folder.
        "HF_MODULES_CACHE": str(home / "hf_modules"),
    })
    return env


def _stream(cmd: list, env: dict, cwd: Path, keep: int = 200) -> tuple[int, list[str]]:
    log.info("running: %s", " ".join(map(str, cmd)))
    proc = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(cwd))
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        log.info("| %s", line)
        tail = (tail + [line])[-keep:]
    return proc.wait(), tail


def _download(fs, path: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(path, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 << 20)


def _ls(fs) -> list[str]:
    try:
        return [f.path for f in fs.ls()]
    except Exception as e:  # empty / brand-new dataset
        if "RunningAsSnapshot" in str(e):  # first build of an incremental output: it starts empty
            log.info("output has no previous files (first build, or Foundry ran this build as "
                     "a snapshot)")
        else:
            log.info("could not list files (%s); assuming none", e)
        return []


def _free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def _pick_scratch() -> str:
    cands = [os.environ.get("TMPDIR"), os.environ.get("USER_WORKING_DIR"), "/tmp", "/work",
             os.getcwd()]
    ok = [c for c in dict.fromkeys(x for x in cands if x)
          if os.path.isdir(c) and os.access(c, os.W_OK)]
    return max(ok, key=_free_gb)


def _version_num(folder: str) -> int:
    m = re.search(r"_v(\d+)$", folder)
    return int(m.group(1)) if m else 1


def _stem_of(folder: str) -> str:
    return re.sub(r"_v\d+$", "", folder)


def _next_version(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    pat = re.compile(rf"^{re.escape(name)}_v(\d+)$")
    taken = [int(m.group(1)) for x in existing if (m := pat.match(x))]
    return f"{name}_v{max(taken + [1]) + 1}"


def _run_key(source_folder: str) -> str:
    """Image code + config, this transform's settings and the result being refined."""
    h = hashlib.sha256()
    for f in sorted(APP_DIR.glob("*.py")) + sorted(APP_DIR.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    h.update(json.dumps(DIFIX_OVERRIDES, sort_keys=True).encode())
    h.update(str(RERUN).encode())
    h.update(source_folder.encode())
    return h.hexdigest()[:16]


def _read_json(fs, path: str) -> dict:
    try:
        with fs.open(path, "rb") as f:
            return json.loads(f.read().decode())
    except Exception:
        return {}


# --------------------------------------------------------------------------- planning
def _candidates(splat_files: list[str]) -> dict[str, list[str]]:
    """Result folders that can be refined: {folder: [its files]} for folders with a splat
    named after the folder and a scene bundle."""
    by_folder: dict[str, list[str]] = {}
    for p in splat_files:
        if "/" in p:
            by_folder.setdefault(p.split("/", 1)[0], []).append(p)
    out = {}
    for folder, files in by_folder.items():
        fs = set(files)
        if f"{folder}/{folder}.ply" in fs and f"{folder}/scene/config.yaml" in fs:
            out[folder] = files
        elif f"{folder}/{folder}.ply" in fs:
            log.info("%s: no scene/ bundle (made before the scene-bundle update); re-run it in "
                     "splat_transform (bump RERUN) to make it refinable", folder)
    return out


def _choose_targets(cands: dict[str, list[str]]) -> list[str]:
    if TARGETS == ["latest"] or TARGETS == "latest":
        newest: dict[str, str] = {}
        for folder in cands:
            stem = _stem_of(folder)
            if stem not in newest or _version_num(folder) > _version_num(newest[stem]):
                newest[stem] = folder
        return sorted(newest.values())
    missing = [t for t in TARGETS if t not in cands]
    if missing:
        raise ValueError(f"TARGETS {missing} are not refinable results in the splats dataset "
                         f"(need <folder>/<folder>.ply and <folder>/scene/). "
                         f"Refinable: {sorted(cands)}")
    return list(TARGETS)


def _find_video(folder: str, splats_fs, video_paths: list[str]) -> str:
    info = _read_json(splats_fs, f"{folder}/{RUN_KEY_FILE}")
    if info.get("video") in video_paths:
        return info["video"]
    stem = _stem_of(folder)
    hits = [p for p in video_paths if Path(p).stem == stem]
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"{folder}: can't find its video in the videos dataset "
                            f"(looked for {info.get('video') or stem + '.*'}; found {hits})")


# --------------------------------------------------------------------------- run
def run_difix(videos, splats, difix_out) -> dict:
    vid_fs = videos.filesystem(mode="current")
    spl_fs = splats.filesystem(mode="current")
    out_fs = difix_out.filesystem()
    video_paths = [p for p in _ls(vid_fs) if Path(p).suffix.lower() in VIDEO_EXTS]
    cands = _candidates(_ls(spl_fs))
    targets = _choose_targets(cands)

    out_files = _ls(out_fs)
    existing = {p.split("/", 1)[0] for p in out_files if "/" in p}
    infos = [_read_json(out_fs, p) for p in out_files
             if p.count("/") == 1 and p.endswith("/" + RUN_KEY_FILE)]
    done_keys = {i.get("run_key") for i in infos if i.get("status") == "ok"}  # failures retry
    todo = []
    for folder in targets:
        key = _run_key(folder)
        if key in done_keys:
            log.info("%s: already refined with this image + settings (run key %s); skipping. "
                     "Bump RERUN to refine it again.", folder, key)
        else:
            todo.append((folder, key))
    log.info("%d target(s) to refine: %s", len(todo), [f for f, _ in todo])
    if not todo:
        return {"refined": [], "failed": []}

    if not Path(DIFIX_PYTHON).exists():
        raise RuntimeError(f"{DIFIX_PYTHON} not found: the image was built without Difix. "
                           "Rebuild it with the default WITH_DIFIX=1.")
    scratch = Path(tempfile.mkdtemp(prefix="difix_", dir=_pick_scratch()))
    home = scratch / "home"
    home.mkdir()
    log.info("scratch %s (%.0f GB free)", scratch, _free_gb(str(scratch)))

    if SELFTEST:
        code, tail = _stream([IMAGE_PYTHON, APP_DIR / "selftest.py", "--only-difix",
                              "--work", scratch / "selftest"], _env(scratch / "selftest", home),
                             scratch)
        if code != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            raise RuntimeError("Difix self-test failed; nothing was refined:\n"
                               + "\n".join(tail[-25:]))

    refined, failed = [], []
    for folder, key in todo:
        job = scratch / folder
        src, out, work = job / "src", job / "out", job / "work"
        out.mkdir(parents=True)
        name = _next_version(f"{folder}_difix", existing)
        existing.add(name)
        status, tail = "failed", []
        try:
            vpath = _find_video(folder, spl_fs, video_paths)
            log.info("%s: downloading splat, scene bundle and video %s", folder, vpath)
            for p in cands[folder]:
                rel = p.split("/", 1)[1]
                if rel == f"{folder}.ply" or rel.startswith("scene/"):
                    _download(spl_fs, p, src / rel)
            video = src / ("video" + Path(vpath).suffix.lower())
            _download(vid_fs, vpath, video)
            code, tail = _stream([IMAGE_PYTHON, APP_DIR / "difix_refine.py",
                                  "--video", video, "--scene", src / "scene",
                                  "--ply", src / f"{folder}.ply", "--out", out, "--work", work,
                                  "--overrides", json.dumps(DIFIX_OVERRIDES)],
                                 _env(work, home), job)
            if code == 0 and (out / "splat.ply").exists():
                (out / "splat.ply").rename(out / f"{name}.ply")
                status = "ok"
            else:
                raise RuntimeError(f"difix_refine.py exited {code}")
        except Exception as e:
            log.error("%s: %s", folder, e)
            (out / "FAILED.txt").write_text(f"Difix refinement of {folder} failed: {e}\n\n"
                                            "Last lines of the log:\n" + "\n".join(tail) + "\n")
        (out / RUN_KEY_FILE).write_text(json.dumps(
            {"source": folder, "run_key": key, "image": IMAGE_NAME, "tag": IMAGE_TAG,
             "overrides": DIFIX_OVERRIDES, "rerun": RERUN, "status": status,
             "note": "Contains AI-generated content (NVIDIA Difix3D+, non-commercial licence)."},
            indent=2))
        for f in sorted(p for p in out.rglob("*") if p.is_file()):
            rel = f"{name}/{f.relative_to(out).as_posix()}"
            with open(f, "rb") as s, out_fs.open(rel, "wb") as d:
                shutil.copyfileobj(s, d, length=16 << 20)
            log.info("uploaded %s (%.1f MB)", rel, f.stat().st_size / 1e6)
        (refined if status == "ok" else failed).append(name)
        shutil.rmtree(job, ignore_errors=True)  # free disk before the next target

    shutil.rmtree(scratch, ignore_errors=True)
    log.info("done: refined %s, failed %s", refined, failed)
    if not refined:
        raise RuntimeError(f"All {len(failed)} target(s) failed; see FAILED.txt and the log.")
    return {"refined": refined, "failed": failed}


@incremental(semantic_version=1)
@transform.using(
    videos=Input(VIDEOS_DATASET),
    splats=Input(SPLATS_DATASET),
    difix_out=Output(DIFIX_DATASET),
).with_resources(
    cpu_cores=CPU_CORES, memory_gb=MEMORY_GB, gpu_type=GPU_TYPE,
).with_container(
    container_image=IMAGE_NAME, container_tag=IMAGE_TAG,
)
def build_difix(videos, splats, difix_out):
    logging.basicConfig(level=logging.INFO)
    run_difix(videos, splats, difix_out)
