"""
Foundry lightweight transform that runs the 360-video -> Gaussian-splat pipeline inside our
own Docker image ("bring your own container") on a GPU, reading videos from one dataset and
writing the finished files straight into another.

Setup (see FOUNDRY.md, "Lightweight transform"):
  1. Build the image with --build-arg APP_UID=5001 (BYOC requires user 5001) and push it to
     an Artifacts repository.
  2. Add that Artifacts repository as a backing repository of this Code Repository.
  3. Put this file under src/<your package>/ (e.g. src/myproject/datasets/splat_transform.py),
     fill in the CONFIG block below, commit, and press Build on the output dataset.

How it behaves
  * Each build looks at every video in the input and processes the ones that have no result
    yet from the CURRENT image + settings. A "run key" (a hash of the pipeline code and
    config inside the image, CONFIG_OVERRIDES and RERUN) is saved with every result, so
    pushing a changed container, changing CONFIG_OVERRIDES or bumping RERUN re-processes
    the videos, while an unchanged setup skips them. Results are appended, never replaced.
  * Output layout: <video name>/<video name>.ply, run summary, pipeline.log, training
    stats and eval/*.jpg photo-vs-splat comparisons.
  * A video that fails writes <video name>/FAILED.txt (with the end of its log) instead of
    a .ply, so one bad video doesn't lose the others. If every video fails, the build fails.
  * Never overwrites: if <video name>/ already exists in the output, the new result goes to
    <video name>_v2/ (then _v3, ...), with the .ply and .json renamed to match, e.g.
    patio_v2/patio_v2.ply. Re-upload a video (or upload it under any name) to get a new version.
"""
import hashlib
import re
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from transforms.api import Input, Output, incremental, transform

# ----------------------------------------------------------------------------- CONFIG
VIDEOS_DATASET = "ri.foundry.main.dataset.9edeb675-87e7-4c9c-bac4-d6ba450db4f7"   # input: raw .mp4 files
SPLATS_DATASET = "ri.foundry.main.dataset.f53697ca-5c35-40ab-b1ae-fec6946b3ac0"  # output: finished splats
IMAGE_NAME = "hello-world"                                # image name in the Artifacts repository
IMAGE_TAG = "0.0.1"                                 # the tag you pushed (never "latest")
CPU_CORES = 8
MEMORY_GB = 32
GPU_TYPE = "NVIDIA_T4"
# Optional config.yaml overrides for this deployment, e.g.
# {"sfm": {"mapper": "mapanything", "mapanything": {"settings": {"window_frames": 24}}}}
CONFIG_OVERRIDES: dict = {}
# Run the image's self-test (~3-5 min on a T4) before processing videos, so a broken image
# fails in minutes instead of hours. SELFTEST_ONLY=True just runs the test (no videos needed)
# and writes the report to _selftest/report.txt in the output dataset.
SELFTEST = True
SELFTEST_ONLY = False
# True: re-process videos whenever the image's code/config or CONFIG_OVERRIDES change (each
# re-run is published as <name>_v2, _v3...). False: only process videos with no result yet.
REPROCESS_ON_CHANGE = True
# Bump this number to force every video to be processed again with the same image/settings.
RERUN = 0
# ------------------------------------------------------------------------------------

# Inside the image (normally no need to change these)
IMAGE_PYTHON = os.environ.get("SPLAT_IMAGE_PYTHON", "/opt/venv/bin/python")
IMAGE_APP = os.environ.get("SPLAT_IMAGE_APP", "/opt/app/foundry_app.py")
IMAGE_SELFTEST = os.environ.get("SPLAT_IMAGE_SELFTEST", "/opt/app/selftest.py")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}

log = logging.getLogger(__name__)


def _pipeline_env(input_dir: Path, output_dir: Path, work_dir: Path, home: Path) -> dict:
    """Environment for the image's own Python: the transform runtime's Python variables
    are removed so the two environments can't mix."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "CONDA", "VIRTUAL_ENV"))}
    venv_bin = str(Path(IMAGE_PYTHON).parent)
    # The transform runtime may put its own libraries on LD_LIBRARY_PATH; those could shadow
    # the image's (libstdc++, libssl...) and break COLMAP/torch. Keep only GPU driver/CUDA dirs.
    gpu_dirs = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if d and ("nvidia" in d or "cuda" in d)]
    lib_path = ":".join(dict.fromkeys(gpu_dirs + [
        "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"]))
    env.update({
        "LD_LIBRARY_PATH": lib_path,
        "PATH": f"{venv_bin}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "VIRTUAL_ENV": str(Path(venv_bin).parent),
        "PYTHONUNBUFFERED": "1",
        "SPLAT_MODE": "local",
        "SPLAT_LOCAL_INPUT": str(input_dir),
        "SPLAT_LOCAL_OUTPUT": str(output_dir),
        "SPLAT_WORK": str(work_dir),
        "SPLAT_KEEP_WORK": "0",
        "COLMAP_BIN": "colmap",
        "HF_HOME": "/opt/hf",
        "HF_HUB_OFFLINE": "1",
        "TORCH_HOME": "/opt/torchhub",
        "HOME": str(home),
        "TORCH_EXTENSIONS_DIR": str(home / "torch_extensions"),
        "SPLAT_CONFIG_OVERRIDES": json.dumps(CONFIG_OVERRIDES),
    })
    return env


def _list_videos(videos):
    fs = videos.filesystem(mode="current")  # every video, not just newly added ones
    return [f.path for f in fs.ls() if Path(f.path).suffix.lower() in VIDEO_EXTS], fs


RUN_KEY_FILE = "run_key.json"


def _run_key() -> str:
    """Fingerprint of what would produce a result: the pipeline code + config baked into the
    image (so re-pushing the same tag with new contents still counts as a change), plus
    this transform's CONFIG_OVERRIDES and RERUN."""
    h = hashlib.sha256()
    app_dir = Path(IMAGE_APP).parent
    for f in sorted(app_dir.glob("*.py")) + sorted(app_dir.glob("*.yaml")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    h.update(json.dumps(CONFIG_OVERRIDES, sort_keys=True).encode())
    h.update(str(RERUN).encode())
    return h.hexdigest()[:16]


def _previous_runs(out_fs) -> tuple[dict[str, set[str]], set[str]]:
    """({video path: {run keys already published}}, {result folders without a run key})."""
    keys: dict[str, set[str]] = {}
    folders, keyed = set(), set()
    try:
        files = [f.path for f in out_fs.ls()]
    except Exception as e:
        log.info("could not list existing output files (%s); assuming none", e)
        return keys, set()
    for path in files:
        parts = path.split("/")
        if len(parts) < 2:
            continue
        folders.add(parts[0])
        if len(parts) == 2 and parts[1] == RUN_KEY_FILE:
            try:
                with out_fs.open(path, "rb") as f:
                    info = json.loads(f.read().decode())
                keys.setdefault(info["video"], set()).add(info["run_key"])
                keyed.add(parts[0])
            except Exception as e:
                log.warning("unreadable %s (%s); ignoring it", path, e)
    return keys, folders - keyed


def _stem_of(folder: str) -> str:
    return re.sub(r"_v\d+$", "", folder)


def _download(fs, path: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with fs.open(path, "rb") as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 << 20)


def _existing_results(out_fs) -> set[str]:
    """Top-level folders already in the output dataset (one per processed video/version)."""
    try:
        return {f.path.split("/", 1)[0] for f in out_fs.ls() if "/" in f.path}
    except Exception as e:  # empty or brand-new output dataset
        log.info("could not list existing output files (%s); assuming none", e)
        return set()


def _next_version(stem: str, existing: set[str]) -> str:
    """'patio' -> 'patio' if unused, else 'patio_v2', 'patio_v3', ... (one past the highest)."""
    if stem not in existing:
        return stem
    pat = re.compile(rf"^{re.escape(stem)}_v(\d+)$")
    taken = [int(m.group(1)) for name in existing if (m := pat.match(name))]
    return f"{stem}_v{max(taken + [1]) + 1}"


def _upload_tree(local_root: Path, out_fs, renames: dict[str, str] | None = None) -> int:
    """Copy local results into the output dataset. `renames` maps a result folder (video
    stem) to the versioned name to publish it under; files named after the stem
    (patio.ply, patio.json) are renamed to match."""
    renames = renames or {}
    n = 0
    for f in sorted(p for p in local_root.rglob("*") if p.is_file()):
        parts = f.relative_to(local_root).parts
        new = renames.get(parts[0], parts[0])
        if new != parts[0]:
            name = parts[-1]
            if len(parts) == 2 and Path(name).stem == parts[0]:
                name = new + Path(name).suffix
            parts = (new, *parts[1:-1], name)
        rel = "/".join(parts)
        with open(f, "rb") as src, out_fs.open(rel, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)
        log.info("uploaded %s (%.1f MB)", rel, f.stat().st_size / 1e6)
        n += 1
    return n


def _free_gb(path: str) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return 0.0


def _pick_scratch(scratch_root):
    """Largest writable temp location: 8K videos need tens of GB of working space."""
    if scratch_root:
        return scratch_root
    cands = [os.environ.get("TMPDIR"), os.environ.get("USER_WORKING_DIR"), "/tmp", "/work",
             os.getcwd()]
    best, best_free = None, -1.0
    for c in dict.fromkeys(x for x in cands if x):
        if os.path.isdir(c) and os.access(c, os.W_OK) and _free_gb(c) > best_free:
            best, best_free = c, _free_gb(c)
    return best


def preflight(scratch_dir: str):
    """Cheap checks before downloading any video, so a broken image fails in seconds with a
    clear message instead of hours later."""
    problems = []
    log.info("user id %s, scratch %s (%.0f GB free)", os.getuid(), scratch_dir,
             _free_gb(scratch_dir))
    if not Path(IMAGE_PYTHON).exists():
        problems.append(f"{IMAGE_PYTHON} not found: is the transform running in the splat image?")
    if not Path(IMAGE_APP).exists():
        problems.append(f"{IMAGE_APP} not found in the image")
    for tool in ("ffmpeg", "ffprobe", "colmap"):
        if not shutil.which(tool, path="/usr/local/bin:/usr/bin:/bin"):
            problems.append(f"'{tool}' not found in the image")
    if problems:
        raise RuntimeError("Image check failed:\n  " + "\n  ".join(problems))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=60)
        log.info("GPU: %s", (out.stdout or out.stderr).strip() or "nvidia-smi gave no output")
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("nvidia-smi unavailable (%s); the pipeline will report if CUDA is missing", e)
    if _free_gb(scratch_dir) < 40:
        log.warning("Only %.0f GB free for working files; a 60 s 8K video can need 30-60 GB. "
                    "Try a short clip first.", _free_gb(scratch_dir))


def _stream(cmd: list, env: dict, cwd: Path, keep: int = 200) -> tuple[int, list[str]]:
    """Run a command in the image's Python, copying its output into the build log."""
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


def run_selftest(scratch: Path) -> tuple[bool, str]:
    work = scratch / "selftest"
    work.mkdir(parents=True, exist_ok=True)
    env = _pipeline_env(scratch / "input", scratch / "output", work, scratch / "home")
    code, tail = _stream([IMAGE_PYTHON, IMAGE_SELFTEST, "--work", work], env, scratch, keep=400)
    shutil.rmtree(work, ignore_errors=True)
    return code == 0, "\n".join(tail)


def run_splat_pipeline(videos, splats, scratch_root: str | None = None) -> dict:
    """The transform body, kept separate from the decorators so it can be tested locally."""
    all_paths, in_fs = _list_videos(videos)
    out_fs = splats.filesystem()
    run_key = _run_key()
    published, legacy = _previous_runs(out_fs)
    legacy_stems = {_stem_of(f) for f in legacy}
    paths = []
    for p in all_paths:
        stem = Path(p).stem
        if REPROCESS_ON_CHANGE:
            if run_key in published.get(p, set()):
                log.info("%s: already processed with this image + settings (run key %s); "
                         "skipping. Bump RERUN to force a new version.", p, run_key)
                continue
        elif p in published or stem in legacy_stems:
            log.info("%s: already has a result; skipping (REPROCESS_ON_CHANGE is False)", p)
            continue
        paths.append(p)
    log.info("run key %s: %d of %d video(s) to process: %s", run_key, len(paths),
             len(all_paths), paths)
    if not paths and not SELFTEST_ONLY:
        return {"processed": 0, "ok": [], "failed": []}

    scratch_root = _pick_scratch(scratch_root)
    preflight(scratch_root)
    scratch = Path(tempfile.mkdtemp(prefix="splat_", dir=scratch_root))
    in_dir, out_dir, work_dir, home = (scratch / d for d in ("input", "output", "work", "home"))
    for d in (in_dir, out_dir, work_dir, home):
        d.mkdir(parents=True)

    if SELFTEST or SELFTEST_ONLY:
        ok_test, report = run_selftest(scratch)
        if SELFTEST_ONLY:
            with splats.filesystem().open("_selftest/report.txt", "wb") as f:
                f.write((report + "\n").encode())
            shutil.rmtree(scratch, ignore_errors=True)
            if not ok_test:
                raise RuntimeError("Self-test failed (see the build log / _selftest/report.txt):\n"
                                   + "\n".join(report.splitlines()[-25:]))
            log.info("Self-test passed (SELFTEST_ONLY: no videos processed)")
            return {"processed": 0, "ok": [], "failed": [], "selftest": "passed"}
        if not ok_test:
            shutil.rmtree(scratch, ignore_errors=True)
            raise RuntimeError("Self-test failed, so no videos were processed. Fix the image and "
                               "rebuild (or set SELFTEST = False to skip the test):\n"
                               + "\n".join(report.splitlines()[-25:]))

    used_names = set()
    source_of: dict[str, str] = {}  # local file stem -> input video path
    for p in paths:
        name = Path(p).name
        if name in used_names:  # same file name in two folders: keep both
            name = p.replace("/", "__")
        used_names.add(name)
        source_of[Path(name).stem] = p
        log.info("downloading %s", p)
        _download(in_fs, p, in_dir / name)

    code, tail = _stream([IMAGE_PYTHON, IMAGE_APP],
                         _pipeline_env(in_dir, out_dir, work_dir, home), scratch)

    ok, failed = [], []
    for name in sorted(used_names):
        stem = Path(name).stem
        if (out_dir / stem / f"{stem}.ply").exists():
            ok.append(stem)
        else:
            failed.append(stem)
            (out_dir / stem).mkdir(parents=True, exist_ok=True)
            (out_dir / stem / "FAILED.txt").write_text(
                f"No splat was produced for {name} (pipeline exit code {code}).\n"
                "Last lines of the build log:\n\n" + "\n".join(tail) + "\n")

    if not ok:
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(f"All {len(failed)} video(s) failed (exit code {code}). "
                           "See the build log above and FAILED.txt.\n" + "\n".join(tail[-30:]))
    for stem in ok + failed:  # remember which video + run key produced each result
        (out_dir / stem / RUN_KEY_FILE).write_text(json.dumps(
            {"video": source_of.get(stem, stem), "run_key": run_key, "image": IMAGE_NAME,
             "tag": IMAGE_TAG, "config_overrides": CONFIG_OVERRIDES, "rerun": RERUN,
             "status": "ok" if stem in ok else "failed"}, indent=2))
    existing = _existing_results(out_fs)
    renames = {}
    for stem in ok + failed:
        renames[stem] = _next_version(stem, existing)
        existing.add(renames[stem])
        if renames[stem] != stem:
            log.info("%s already exists in the output: publishing this run as %s",
                     stem, renames[stem])
    n = _upload_tree(out_dir, out_fs, renames)
    log.info("done: %d ok %s, %d failed %s, %d files written",
             len(ok), [renames[s] for s in ok], len(failed), [renames[s] for s in failed], n)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"processed": len(paths), "ok": [renames[s] for s in ok],
            "failed": [renames[s] for s in failed]}


@incremental(semantic_version=1)
@transform.using(
    videos=Input(VIDEOS_DATASET),
    splats=Output(SPLATS_DATASET),
).with_resources(
    cpu_cores=CPU_CORES, memory_gb=MEMORY_GB, gpu_type=GPU_TYPE,
).with_container(
    container_image=IMAGE_NAME, container_tag=IMAGE_TAG,
)
def build_splats(videos, splats):
    logging.basicConfig(level=logging.INFO)
    run_splat_pipeline(videos, splats)