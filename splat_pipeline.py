#!/usr/bin/env python3
"""
splat_pipeline.py - 360 degree equirectangular video -> Gaussian splat (.ply)

Stages (each one is resumable):
    extract  ffmpeg pulls frames from the video at extract.fps
    select   keeps the sharpest frame in each window -> select.target_fps
    slice    equirect frame -> pinhole views: a cube map or a custom view list, written
             as images/vNN/<frame>.jpg (+ optional nadir masks)
    sfm      COLMAP features + matching + mapper. The views of one 360 frame are declared
             as a camera rig (shared centre, known rotations), so COLMAP solves one pose
             per frame instead of one per view
    gate     rejects captures with too few registered images / no parallax
    train    runs the configured splat trainer (Nerfstudio splatfacto or gsplat)
    export   finds / exports the .ply and copies it to the output folder

Usage:
    python splat_pipeline.py room.mp4 --config config.yaml --work work --out out
    python splat_pipeline.py videos_folder/ --config config.yaml
    python splat_pipeline.py room.mp4 --from-stage train      # re-run from a stage
    python splat_pipeline.py room.mp4 --until-stage gate      # just check the capture

Environment variables (useful in containers): SPLAT_CONFIG, SPLAT_WORK, SPLAT_OUT
set the defaults for --config, --work and --out. Other code can call
run_pipeline() directly instead of using the command line.

Input must already be stitched equirectangular (2:1) video. Raw dual-fisheye
files (.insv, .360 etc.) need stitching in the camera vendor's software first.

Python deps: numpy, opencv-python, pyyaml
External:    ffmpeg/ffprobe, colmap, and a trainer. The default trainer is
             train_gsplat.py next to this file (needs torch + gsplat, see README).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import naming  # noqa: E402

STAGES = ["extract", "select", "slice", "sfm", "gate", "train", "export"]
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
PAD = 2  # horizontal wrap padding (px) so remap interpolates across the 360 seam

log = logging.getLogger("splat")


class PipelineError(RuntimeError):
    pass


class GateError(PipelineError):
    pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: dict = {
    "extract": {"fps": 6, "max_frames": 0, "keep_raw": False},
    "select": {"target_fps": 2, "min_sharpness": 0.0},
    "slice": {
        "layout": "cubemap",       # "cubemap" (6 faces) or "custom" (the views list below)
        "cubemap_faces": None,     # subset of front/right/back/left/up/down; None = all six
        "size": 1600,
        "fov_deg": 90,
        "views": [[y, 0] for y in range(0, 360, 45)]
        + [[y, 45] for y in range(0, 360, 90)]
        + [[0, 90]],
        "nadir_mask_deg": None,
        "jpeg_quality": 95,
        "workers": 8,
    },
    "sfm": {
        "colmap_bin": "colmap",
        "mapper": "colmap",
        "auto_min_registered": 0.9,
        "auto_global_timeout_min": 20,
        "glomap_bin": "glomap",
        "matcher": "sequential",
        "sequential_overlap_frames": 8,
        "rig": True,             # treat the views of each 360 frame as one camera rig
        "use_gpu": True,
        "max_features": 8192,
        "fix_intrinsics": True,
        "mapanything": {
            "python": "",            # interpreter with MapAnything installed ("" = this one)
            "fallback": "colmap",    # mapper to use if the MapAnything route fails ("" = fail)
            "refine_rounds": 2,      # COLMAP triangulate + bundle-adjust passes
            "min_observations": 30,  # images with fewer triangulated points are dropped
            "settings": {},          # passed to mapanything_init.py (model, views, windows...)
        },
    },
    "gate": {"min_registered_ratio": 0.7, "min_parallax_ratio": 0.05},
    "train": {
        "backend": "custom",
        "iterations": 30000,
        "vars": {},
        "custom": {},
        "commands": {
            "custom": ["{python}", "{script_dir}/train_gsplat.py", "--data", "{data}",
                       "--out", "{train_dir}", "--iterations", "{iterations}",
                       "--config", "{config_path}"],
        },
    },
    "export": {"commands": {}},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict:
    if path is None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    else:
        with open(path) as f:
            cfg = deep_merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})
    cfg["_config_path"] = str(Path(path).resolve()) if path else ""
    return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd: list, cwd: Path | None = None, echo: bool = False,
        timeout: float | None = None) -> str:
    """Run a command, stream output into the job log, raise with a tail on failure (or when it
    runs longer than `timeout` seconds: it is killed)."""
    cmd = [str(c) for c in cmd]
    log.info("$ %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1", "SPLAT_CHILD": "1"},
            start_new_session=bool(timeout),  # own process group, so a timeout kills it all
        )
    except FileNotFoundError:
        raise PipelineError(f"Executable not found: {cmd[0]} (is it installed and on PATH?)")
    tail: list[str] = []
    out_lines: list[str] = []
    timer, timed_out = None, []
    if timeout:
        import threading

        def _kill():
            if proc.poll() is not None:  # finished right at the deadline: not a timeout
                return
            timed_out.append(True)
            try:
                os.killpg(proc.pid, 9)
            except (OSError, AttributeError):  # Windows has no process groups here
                proc.kill()
        timer = threading.Timer(timeout, _kill)
        timer.daemon = True
        timer.start()
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            log.log(logging.INFO if echo else logging.DEBUG, "  %s", line)
            out_lines.append(line)
            tail = (tail + [line])[-40:]
        proc.wait()
    finally:
        if timer:
            timer.cancel()
    if timed_out:
        raise PipelineError(f"{Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''} took longer "
                            f"than {timeout / 60:.0f} min and was stopped")
    if proc.returncode != 0:
        raise PipelineError(
            f"Command failed (exit {proc.returncode}): {cmd[0]}\n--- last output ---\n"
            + "\n".join(tail)
        )
    return "\n".join(out_lines)


def fill_template(template: list, values: dict) -> list[str]:
    try:
        return [str(part).format(**values) for part in template]
    except KeyError as e:
        raise PipelineError(f"Command template uses unknown placeholder {e}; add it under train.vars")


def reset_dir(d: Path) -> Path:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def tool_help(binary: str, sub: str) -> str:
    try:
        r = subprocess.run([binary, sub, "--help"], capture_output=True, text=True, timeout=60)
        return r.stdout + r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, video: Path, work_root: Path, out_root: Path, cfg: dict):
        self.video = video.resolve()
        self.name = video.stem
        self.dir = (work_root / self.name).resolve()
        self.out = out_root.resolve()
        self.cfg = cfg
        self.markers = self.dir / ".stages"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.markers.mkdir(exist_ok=True)

    # paths
    raw = property(lambda s: s.dir / "raw")
    frames = property(lambda s: s.dir / "frames")
    images = property(lambda s: s.dir / "images")
    masks = property(lambda s: s.dir / "masks")          # nadir masks (SfM + training)
    sfm_masks = property(lambda s: s.dir / "sfm_masks")  # + feature ownership (SfM only)
    db = property(lambda s: s.dir / "database.db")
    sparse_raw = property(lambda s: s.dir / "sparse_raw")
    sparse = property(lambda s: s.dir / "sparse")
    train_dir = property(lambda s: s.dir / "train")
    export_dir = property(lambda s: s.dir / "export")

    def done(self, stage: str) -> bool:
        return (self.markers / f"{stage}.done").exists()

    def mark(self, stage: str, info: dict | None = None):
        (self.markers / f"{stage}.done").write_text(json.dumps(info or {}, indent=2))

    def invalidate_from(self, stage: str):
        for s in STAGES[STAGES.index(stage):]:
            (self.markers / f"{s}.done").unlink(missing_ok=True)

    def read_json(self, name: str) -> dict:
        return json.loads((self.dir / name).read_text())

    def write_json(self, name: str, data: dict):
        (self.dir / name).write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- #
# Stage: extract
# --------------------------------------------------------------------------- #
def stage_extract(job: Job):
    c = job.cfg["extract"]
    probe = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", job.video])
    w, h = (int(x) for x in probe.strip().splitlines()[0].split(",")[:2])
    log.info("Video resolution %dx%d", w, h)
    if abs(w / h - 2.0) > 0.02:
        raise PipelineError(
            f"Video is {w}x{h}, not 2:1. Expected stitched equirectangular footage "
            "(stitch dual-fisheye files in the camera's software first)."
        )
    out = reset_dir(job.raw)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", job.video,
           "-vf", f"fps={c['fps']}", "-q:v", "2"]
    if c.get("max_frames"):
        cmd += ["-frames:v", c["max_frames"]]
    cmd.append(out / "%06d.jpg")
    run(cmd)
    n = len(list(out.glob("*.jpg")))
    if n == 0:
        raise PipelineError("ffmpeg produced no frames")
    log.info("Extracted %d frames", n)
    job.write_json("video.json", {"width": w, "height": h, "extracted": n})


# --------------------------------------------------------------------------- #
# Stage: select (sharpest frame per window)
# --------------------------------------------------------------------------- #
def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        return 0.0
    h = img.shape[0]
    band = img[h // 4: 3 * h // 4]  # skip poles: stretched and often the operator/sky
    return float(cv2.Laplacian(band, cv2.CV_64F).var())


def stage_select(job: Job):
    c, ec = job.cfg["select"], job.cfg["extract"]
    raw = sorted(job.raw.glob("*.jpg"))
    window = max(1, round(ec["fps"] / c["target_fps"]))
    with ThreadPoolExecutor(job.cfg["slice"]["workers"]) as ex:
        scores = list(ex.map(sharpness, raw))
    out = reset_dir(job.frames)
    kept, dropped_blurry = 0, 0
    chosen = {}  # selected frame -> extracted frame file, so the selection can be replayed
    for i in range(0, len(raw), window):
        chunk = list(range(i, min(i + window, len(raw))))
        best = max(chunk, key=lambda k: scores[k])
        if scores[best] < c["min_sharpness"]:
            dropped_blurry += 1
            continue
        shutil.copy2(raw[best], out / f"f{kept:05d}.jpg")
        chosen[f"f{kept:05d}"] = raw[best].name
        kept += 1
    job.write_json("selection.json", {"extract_fps": ec["fps"], "frames": chosen})
    log.info("Selected %d frames (window=%d, %d windows too blurry); sharpness median %.1f",
             kept, window, dropped_blurry, float(np.median(scores)) if scores else 0)
    if kept < 10:
        raise PipelineError(f"Only {kept} usable frames. Capture longer or lower select.min_sharpness.")
    if not ec.get("keep_raw"):
        shutil.rmtree(job.raw, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Stage: slice (equirect -> pinhole views)
# --------------------------------------------------------------------------- #
def view_dirs(size: int, fov_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit ray per pixel of a view, in the level 360 frame (x right, y down, z forward)."""
    f = 0.5 * size / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    d = np.stack([(u - size / 2) / f, (v - size / 2) / f, np.ones((size, size))], -1)
    d = d @ naming.rig_from_cam(yaw_deg, pitch_deg).T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ownership_masks(views: list, size: int, fov_deg: float) -> list[np.ndarray]:
    """For overlapping views: each 360 pixel is 'owned' by the view whose axis is closest,
    so SfM extracts its features once rather than in every view that sees it (the same
    trick as COLMAP's panorama example). Training still uses every pixel."""
    axes = np.array([naming.rig_from_cam(*v[:2]) @ np.array([0, 0, 1.0]) for v in views])
    out = []
    for i, (yaw, pitch) in enumerate(views):
        owner = np.argmax(view_dirs(size, fov_deg, yaw, pitch) @ axes.T, axis=-1)
        out.append((owner == i).astype(np.uint8) * 255)
    return out


def build_view_maps(eq_w: int, eq_h: int, size: int, fov_deg: float,
                    yaw_deg: float, pitch_deg: float):
    """Remap tables for one pinhole view. Camera: x right, y down, z forward.
    Positive yaw turns right, positive pitch looks up."""
    f = 0.5 * size / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    x = (u - size / 2) / f
    y = (v - size / 2) / f
    z = np.ones_like(x)
    p, t = math.radians(pitch_deg), math.radians(yaw_deg)
    y2 = y * math.cos(p) - z * math.sin(p)
    z2 = y * math.sin(p) + z * math.cos(p)
    x3 = x * math.cos(t) + z2 * math.sin(t)
    z3 = -x * math.sin(t) + z2 * math.cos(t)
    lon = np.arctan2(x3, z3)
    lat = -np.arctan2(y2, np.hypot(x3, z3))
    map_x = (lon / (2 * np.pi) + 0.5) * eq_w - 0.5 + PAD
    map_y = np.clip((0.5 - lat / np.pi) * eq_h - 0.5, 0, eq_h - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32), np.degrees(lat), f


def stage_slice(job: Job):
    c = job.cfg["slice"]
    frames = sorted(job.frames.glob("*.jpg"))
    first = cv2.imread(str(frames[0]))
    eq_h, eq_w = first.shape[:2]
    size, fov = c["size"], c["fov_deg"]
    view_list = naming.resolve_views(c)
    native = eq_w * fov / 360
    if size > native * 1.05:
        log.warning("slice.size=%d exceeds native resolution (~%d px for %d deg FOV); "
                    "views will be upsampled", size, native, fov)
    overlapping = naming.views_overlap(view_list, fov)
    own = ownership_masks(view_list, size, fov) if overlapping else [None] * len(view_list)

    views = []
    for idx, (yaw, pitch) in enumerate(view_list):
        mx, my, lat, f = build_view_maps(eq_w, eq_h, size, fov, yaw, pitch)
        mask = None
        if c.get("nadir_mask_deg") is not None:
            m = (lat >= c["nadir_mask_deg"]).astype(np.uint8) * 255
            if m.min() == 0:
                if m.mean() < 255 * 0.25:
                    log.warning("View %d (yaw %s, pitch %s) is >75%% masked; consider removing it",
                                idx, yaw, pitch)
                mask = m
        sfm_mask = mask
        if own[idx] is not None:
            sfm_mask = own[idx] if mask is None else np.minimum(own[idx], mask)
        views.append((idx, mx, my, mask, sfm_mask))

    for d in (job.images, job.masks, job.sfm_masks):
        if d.exists():
            shutil.rmtree(d)
    job.images.mkdir(parents=True)
    has_mask = any(v[3] is not None for v in views)
    has_sfm_mask = any(v[4] is not None for v in views) and overlapping
    for idx, *_ in views:
        (job.images / naming.view_folder(idx)).mkdir()
        if has_mask:
            (job.masks / naming.view_folder(idx)).mkdir(parents=True)
        if has_sfm_mask:
            (job.sfm_masks / naming.view_folder(idx)).mkdir(parents=True)

    q = [cv2.IMWRITE_JPEG_QUALITY, int(c["jpeg_quality"])]
    full = np.full((size, size), 255, np.uint8)

    def process(frame: Path):
        eq = cv2.imread(str(frame))
        if eq is None or eq.shape[:2] != (eq_h, eq_w):
            raise PipelineError(f"Bad frame {frame}")
        eq = cv2.copyMakeBorder(eq, 0, 0, PAD, PAD, cv2.BORDER_WRAP)
        for idx, mx, my, mask, sfm_mask in views:
            name = naming.image_name(frame.stem, idx)
            img = cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(str(job.images / name), img, q)
            if has_mask:  # COLMAP mask naming: <mask dir>/<image name>.png
                cv2.imwrite(str(job.masks / f"{name}.png"), mask if mask is not None else full)
            if has_sfm_mask:
                cv2.imwrite(str(job.sfm_masks / f"{name}.png"),
                            sfm_mask if sfm_mask is not None else full)

    with ThreadPoolExecutor(c["workers"]) as ex:
        list(ex.map(process, frames))

    f = 0.5 * size / math.tan(math.radians(fov) / 2)
    intr = {"model": "PINHOLE", "width": size, "height": size,
            "fx": f, "fy": f, "cx": size / 2, "cy": size / 2,
            "views_per_frame": len(views), "frames": len(frames),
            "views": [list(v) for v in view_list], "layout": c.get("layout", "custom"),
            "naming": "view_folders", "overlapping_views": overlapping}
    job.write_json("intrinsics.json", intr)
    log.info("Wrote %d images (%d frames x %d views, %s layout, %dpx, f=%.1f%s)",
             len(frames) * len(views), len(frames), len(views), intr["layout"], size, f,
             ", SfM feature-ownership masks" if has_sfm_mask else "")


# --------------------------------------------------------------------------- #
# Stage: sfm
# --------------------------------------------------------------------------- #
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_model_txt(d: Path):
    """Return (camera_centers Nx3, image_names, points Mx3) from a COLMAP TXT model."""
    centers, names = [], []
    lines = [l for l in (d / "images.txt").read_text().splitlines(keepends=True)
             if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        q = list(map(float, parts[1:5]))
        t = np.array(list(map(float, parts[5:8])))
        centers.append(-qvec2rotmat(q).T @ t)
        names.append(parts[9])
    pts = []
    for line in (d / "points3D.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            pts.append(list(map(float, line.split()[1:4])))
    return np.array(centers).reshape(-1, 3), names, np.array(pts).reshape(-1, 3)


def model_stats(centers: np.ndarray, pts: np.ndarray) -> dict:
    if len(centers) == 0 or len(pts) == 0:
        return {"registered": len(centers), "points": len(pts), "parallax_ratio": 0.0}
    centroid = centers.mean(0)
    path_extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    scene_depth = float(np.median(np.linalg.norm(pts - centroid, axis=1)))
    return {
        "registered": int(len(centers)),
        "points": int(len(pts)),
        "camera_path_extent": path_extent,
        "median_scene_depth": scene_depth,
        "parallax_ratio": path_extent / scene_depth if scene_depth > 0 else 0.0,
    }


def resolve_colmap(binary: str) -> str:
    """On Windows, COLMAP ships a COLMAP.bat launcher that breaks when arguments
    contain spaces. Call bin/colmap.exe directly and replicate what the .bat
    does (put lib/ on PATH, set the Qt plugin path) instead."""
    p = Path(binary)
    if p.suffix.lower() != ".bat":
        return binary
    root = p.parent
    exe = root / "bin" / "colmap.exe"
    if not exe.exists():
        log.warning("Could not find %s; falling back to the .bat launcher", exe)
        return binary
    extra = [str(d) for d in (root / "lib", root / "bin") if d.exists()]
    os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])
    plugins = root / "lib" / "plugins"
    if plugins.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
    log.info("Using %s", exe)
    return str(exe)


def pick_model(job: Job, colmap: str):
    """The largest sub-model in sparse_raw (mappers can split a capture into pieces):
    (path, stats, number of models, TXT root)."""
    models = sorted(p for p in job.sparse_raw.iterdir() if p.is_dir()) \
        if job.sparse_raw.exists() else []
    if not models:
        raise GateError("SfM produced no model at all. Classic cause: camera didn't move "
                        "(no parallax), or too few overlapping frames.")
    best, best_stats = None, None
    txt_root = reset_dir(job.dir / "sparse_txt")
    for m in models:
        txt = txt_root / m.name
        txt.mkdir()
        run([colmap, "model_converter", "--input_path", m, "--output_path", txt,
             "--output_type", "TXT"])
        centers, _, pts = read_model_txt(txt)
        stats = model_stats(centers, pts)
        log.info("Model %s: %d images, %d points", m.name, stats["registered"], stats["points"])
        if best_stats is None or stats["registered"] > best_stats["registered"]:
            best, best_stats = m, stats
    return best, best_stats, len(models), txt_root


def auto_mapper(job: Job, colmap: str, c: dict, rig: bool) -> str:
    """Global mapper first (all poses at once: minutes instead of the best part of an hour
    for ~1300 images), falling back to the incremental mapper if it registers too few
    images, fails, or runs past auto_global_timeout_min (it once hung on a 214-frame rig).
    Returns the mapper whose result is in sparse_raw."""
    want = float(c.get("auto_min_registered", 0.9))
    limit = float(c.get("auto_global_timeout_min", 20)) * 60
    total = max(1, len(list(job.images.rglob("*.jpg"))))
    if "GlobalMapper." in tool_help(colmap, "global_mapper"):
        t0 = time.time()
        try:
            run_mapper(job, colmap, c, "colmap_global", rig=rig, timeout=limit)
            _, stats, _, _ = pick_model(job, colmap)
            ratio = stats["registered"] / total
            if ratio >= want:
                log.info("Global mapper: %d/%d images in %.0f min", stats["registered"], total,
                         (time.time() - t0) / 60)
                return "colmap_global"
            log.warning("Global mapper registered only %.0f%% of images (want %.0f%%); "
                        "running the incremental mapper instead", 100 * ratio, 100 * want)
        except (PipelineError, GateError) as e:
            log.warning("Global mapper failed (%s); running the incremental mapper", e)
        reset_dir(job.sparse_raw)
    else:
        log.info("This COLMAP has no global_mapper; using the incremental mapper")
    run_mapper(job, colmap, c, "colmap", rig=rig)
    return "colmap"


def run_mapper(job: Job, colmap: str, c: dict, mapper: str, rig: bool = False,
               timeout: float | None = None):
    if mapper == "colmap_global":
        # GLOMAP built into COLMAP 4.x: solves all poses at once, far faster than incremental
        cmd = [colmap, "global_mapper", "--database_path", job.db,
               "--image_path", job.images, "--output_path", job.sparse_raw]
        g_help = tool_help(colmap, "global_mapper")
        if rig and "GlobalMapper.refine_sensor_from_rig" in g_help:
            cmd += ["--GlobalMapper.refine_sensor_from_rig", "0"]
        if c["fix_intrinsics"]:
            for opt in ("ba_refine_focal_length", "ba_refine_principal_point",
                        "ba_refine_extra_params"):
                if f"GlobalMapper.{opt}" in g_help:
                    cmd += [f"--GlobalMapper.{opt}", "0"]
        run(cmd, timeout=timeout)
    elif mapper == "glomap":
        run([c["glomap_bin"], "mapper", "--database_path", job.db,
             "--image_path", job.images, "--output_path", job.sparse_raw])
    elif mapper == "colmap":
        cmd = [colmap, "mapper", "--database_path", job.db, "--image_path", job.images,
               "--output_path", job.sparse_raw]
        if c["fix_intrinsics"]:
            cmd += ["--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0"]
        if rig:  # the views' relative rotations are exact: keep them fixed
            cmd += ["--Mapper.ba_refine_sensor_from_rig", "0"]
        run(cmd)
    else:
        raise PipelineError(f"Unknown sfm.mapper '{mapper}'")


# --------------------------------------------------------------------------- #
# MapAnything initialisation + COLMAP refinement ("mapanything" mapper)
# --------------------------------------------------------------------------- #
def write_init_model(path: Path, db: Path, intr: dict, poses: dict) -> int:
    """COLMAP TXT model with MapAnything poses, using the database's own image/camera ids
    (point_triangulator matches images to the database by name)."""
    con = sqlite3.connect(str(db))
    try:
        images = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        cams = con.execute("SELECT camera_id, width, height FROM cameras").fetchall()
    finally:
        con.close()
    path.mkdir(parents=True, exist_ok=True)
    cam_lines = [f"{cid} PINHOLE {w} {h} {intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']}"
                 for cid, w, h in cams]
    (path / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    lines, n = [], 0
    for iid, name, cid in images:
        p = poses.get(name)
        if p is None:
            continue
        q, t = p["qvec"], p["tvec"]
        lines.append(f"{iid} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cid} {name}")
        lines.append("")
        n += 1
    (path / "images.txt").write_text("\n".join(lines) + "\n")
    (path / "points3D.txt").write_text("")
    return n


def observations_per_image(txt_dir: Path) -> dict:
    lines = [l for l in (txt_dir / "images.txt").read_text().splitlines()
             if not l.startswith("#")]
    out = {}
    for i in range(0, len(lines) - 1, 2):
        head = lines[i].split()
        if len(head) < 10:
            continue
        obs = lines[i + 1].split()
        out[head[9]] = sum(1 for k in range(2, len(obs), 3) if obs[k] != "-1")
    return out


def frame_centres(txt_dir: Path) -> dict:
    """Mean camera centre per 360 frame (all its views share one physical position)."""
    lines = [l for l in (txt_dir / "images.txt").read_text().splitlines()
             if not l.startswith("#")]
    acc: dict[str, list] = {}
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        R = qvec2rotmat(list(map(float, p[1:5])))
        c = -R.T @ np.array(list(map(float, p[5:8])))
        parsed = naming.parse(p[9])
        acc.setdefault(parsed[0] if parsed else p[9], []).append(c)
    return {f: np.mean(cs, axis=0) for f, cs in acc.items()}


def refine_with_colmap(job: Job, colmap: str, c: dict, intr: dict) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mapanything_init as mai
    ma = c.get("mapanything", {})
    ma_dir = job.dir / "ma_init"
    py = ma.get("python") or sys.executable
    if ma.get("python") and not Path(py).is_absolute():
        cand = Path(__file__).resolve().parent / py
        py = str(cand) if cand.exists() else py

    # 1. Metric poses from MapAnything
    reset_dir(ma_dir)
    run([py, Path(__file__).resolve().parent / "mapanything_init.py", "--data", job.dir,
         "--out", ma_dir, "--settings", json.dumps(ma.get("settings", {}))], echo=True)
    poses = json.loads((ma_dir / "poses.json").read_text())
    ma_report = json.loads((ma_dir / "report.json").read_text())
    rigs = {f: np.array(v["center"]) for f, v in
            json.loads((ma_dir / "rigs.json").read_text()).items()}

    # 2. Seed model with those poses
    work = reset_dir(job.dir / "sfm_refine")
    n = write_init_model(work / "init", job.db, intr, poses)
    if n < 2:
        raise PipelineError("No MapAnything poses matched the database images")
    log.info("Seeded COLMAP with %d MapAnything poses", n)

    # 3. Triangulate with poses fixed, then bundle-adjust; repeat
    tri_help = tool_help(colmap, "point_triangulator")
    ba_help = tool_help(colmap, "bundle_adjuster")
    cur = work / "init"
    for r in range(int(ma.get("refine_rounds", 2))):
        tri = work / f"tri{r}"
        tri.mkdir()
        cmd = [colmap, "point_triangulator", "--database_path", job.db,
               "--image_path", job.images, "--input_path", cur, "--output_path", tri]
        if "refine_intrinsics" in tri_help:
            cmd += ["--refine_intrinsics", "0"]
        run(cmd)
        ba = work / f"ba{r}"
        ba.mkdir()
        cmd = [colmap, "bundle_adjuster", "--input_path", tri, "--output_path", ba]
        if c["fix_intrinsics"]:
            for opt in ("refine_focal_length", "refine_principal_point", "refine_extra_params"):
                if f"BundleAdjustment.{opt}" in ba_help:
                    cmd += [f"--BundleAdjustment.{opt}", "0"]
        run(cmd)
        cur = ba

    # 4. Drop images COLMAP couldn't constrain (they'd only carry MapAnything's estimate)
    txt = work / "txt"
    txt.mkdir()
    run([colmap, "model_converter", "--input_path", cur, "--output_path", txt,
         "--output_type", "TXT"])
    obs = observations_per_image(txt)
    min_obs = int(ma.get("min_observations", 30))
    weak = sorted(name for name, k in obs.items() if k < min_obs)
    if len(weak) > 0.9 * max(len(obs), 1):
        raise PipelineError(f"COLMAP could constrain only {len(obs) - len(weak)}/{len(obs)} "
                            "images with the MapAnything poses")
    if weak:
        (work / "weak.txt").write_text("\n".join(weak) + "\n")
        pruned = work / "pruned"
        pruned.mkdir()
        run([colmap, "image_deleter", "--input_path", cur, "--output_path", pruned,
             "--image_names_path", work / "weak.txt"])
        cur = pruned
        shutil.rmtree(txt)
        txt.mkdir()
        run([colmap, "model_converter", "--input_path", cur, "--output_path", txt,
             "--output_type", "TXT"])
    log.info("Refinement kept %d images (%d dropped with < %d triangulated points)",
             len(obs) - len(weak), len(weak), min_obs)

    # 5. Restore metric scale: map the refined model onto MapAnything's metric frame centres
    refined = frame_centres(txt)
    shared = [f for f in refined if f in rigs]
    src = np.array([refined[f] for f in shared])
    dst = np.array([rigs[f] for f in shared])
    if len(shared) >= 3 and np.linalg.norm(src - src.mean(0), axis=1).max() > 1e-6:
        sc, R, t = mai.umeyama(src, dst)
    else:
        sc, R, t = 1.0, np.eye(3), np.zeros(3)
    resid = np.sqrt(np.mean(np.sum((sc * src @ R.T + t - dst) ** 2, axis=1))) if shared else 0.0
    q = mai.rotmat_to_qvec(R)
    (work / "to_metric.txt").write_text(" ".join(f"{v:.12g}" for v in [sc, *q, *t]) + "\n")
    out = job.sparse_raw / "0"
    out.mkdir(parents=True, exist_ok=True)
    run([colmap, "model_transformer", "--input_path", cur, "--output_path", out,
         "--transform_path", work / "to_metric.txt"])
    log.info("Metric alignment: scale x%.4f, refined vs MapAnything camera positions differ "
             "by %.3f m RMS over %d frames", sc, resid, len(shared))
    return {**ma_report, "refined_images": len(obs) - len(weak), "dropped_weak": len(weak),
            "metric_scale_factor": round(float(sc), 5),
            "colmap_vs_mapanything_rms_m": round(float(resid), 4)}


def stage_sfm(job: Job):
    c = job.cfg["sfm"]
    intr = job.read_json("intrinsics.json")
    colmap = resolve_colmap(c["colmap_bin"])
    if job.train_dir.exists():  # new cameras/points: any old training checkpoint is stale
        shutil.rmtree(job.train_dir)
    job.db.unlink(missing_ok=True)

    ext_help = tool_help(colmap, "feature_extractor")
    if not ext_help:
        raise PipelineError(f"Cannot run '{colmap}'. Install COLMAP or set sfm.colmap_bin.")
    # COLMAP renamed Sift* GPU options to Feature* in newer releases; detect which we have.
    ext_pre = "FeatureExtraction" if "FeatureExtraction.use_gpu" in ext_help else "SiftExtraction"
    gpu = "1" if c["use_gpu"] else "0"

    # Rig mode: one camera per view folder, then tell COLMAP the folders are sensors of one
    # rig (same centre, known rotations). Needs COLMAP >= 3.12 and the vNN/ folder layout.
    folders = intr.get("naming") == "view_folders"
    rig = bool(c.get("rig", True)) and folders and c["mapper"] != "mapanything"
    if rig and "rig_config_path" not in tool_help(colmap, "rig_configurator"):
        log.warning("This COLMAP has no rig support (needs 3.12+); solving views independently")
        rig = False
    if not rig and folders and not intr.get("overlapping_views", True) and \
            c["mapper"] != "mapanything":
        log.warning("Non-overlapping views (cube map) without a rig: COLMAP can't link the faces "
                    "of one frame and will register far fewer images. Enable sfm.rig (COLMAP "
                    "3.12+) or use slice.layout: custom.")
    cmd = [colmap, "feature_extractor", "--database_path", job.db, "--image_path", job.images,
           "--ImageReader.camera_model", "PINHOLE",
           "--ImageReader.camera_params",
           f"{intr['fx']},{intr['fy']},{intr['cx']},{intr['cy']}",
           f"--{ext_pre}.use_gpu", gpu]
    cmd += (["--ImageReader.single_camera_per_folder", "1"] if folders
            else ["--ImageReader.single_camera", "1"])
    if "SiftExtraction.max_num_features" in ext_help:
        cmd += ["--SiftExtraction.max_num_features", c["max_features"]]
    mask_dir = job.sfm_masks if job.sfm_masks.exists() else job.masks
    if mask_dir.exists():
        cmd += ["--ImageReader.mask_path", mask_dir]
    run(cmd)

    if rig:
        rig_cfg = job.dir / "rig_config.json"
        rig_cfg.write_text(json.dumps(naming.colmap_rig_config(intr["views"]), indent=2))
        run([colmap, "rig_configurator", "--database_path", job.db,
             "--rig_config_path", rig_cfg])
        log.info("Configured a %d-view camera rig: one pose per 360 frame", len(intr["views"]))

    def matcher_flags(m_help: str) -> list:
        m_pre = "FeatureMatching" if "FeatureMatching.use_gpu" in m_help else "SiftMatching"
        flags = [f"--{m_pre}.use_gpu", gpu]
        if rig:
            # Views of one frame share a centre (pure rotation): nothing to learn from
            # matching them; and verify matches against the rig geometry.
            for opt in ("rig_verification", "skip_image_pairs_in_same_frame"):
                if f"FeatureMatching.{opt}" in m_help:
                    flags += [f"--FeatureMatching.{opt}", "1"]
        return flags

    if c["matcher"] == "sequential":
        m_help = tool_help(colmap, "sequential_matcher")
        if rig:  # overlap counts frames; each image is matched to every view of those frames
            overlap = c["sequential_overlap_frames"]
        else:
            overlap = intr["views_per_frame"] * c["sequential_overlap_frames"]
        run([colmap, "sequential_matcher", "--database_path", job.db,
             "--SequentialMatching.overlap", overlap,
             "--SequentialMatching.quadratic_overlap", "1", *matcher_flags(m_help)])
    elif c["matcher"] == "exhaustive":
        m_help = tool_help(colmap, "exhaustive_matcher")
        run([colmap, "exhaustive_matcher", "--database_path", job.db, *matcher_flags(m_help)])
    else:
        raise PipelineError(f"Unknown sfm.matcher '{c['matcher']}'")

    reset_dir(job.sparse_raw)
    extra_stats = {"mapper": c["mapper"], "rig": rig}
    if c["mapper"] == "mapanything":
        ma = c.get("mapanything", {})
        try:
            extra_stats["mapanything"] = refine_with_colmap(job, colmap, c, intr)
            extra_stats["metric_scale"] = True
        except PipelineError as e:
            fb = ma.get("fallback") or ""
            if not fb:
                raise
            log.warning("MapAnything route failed (%s). Falling back to the '%s' mapper.", e, fb)
            reset_dir(job.sparse_raw)
            fb_rig = bool(c.get("rig", True)) and folders and \
                "rig_config_path" in tool_help(colmap, "rig_configurator")
            if fb_rig:  # matches are already in the database; add the rig before mapping
                rig_cfg = job.dir / "rig_config.json"
                rig_cfg.write_text(json.dumps(naming.colmap_rig_config(intr["views"]), indent=2))
                run([colmap, "rig_configurator", "--database_path", job.db,
                     "--rig_config_path", rig_cfg])
            run_mapper(job, colmap, c, fb, rig=fb_rig)
            extra_stats["rig"] = fb_rig
            extra_stats.update(mapper=fb, mapanything_failed=str(e)[:500])
    elif c["mapper"] == "auto":
        extra_stats["mapper"] = auto_mapper(job, colmap, c, rig)
    else:
        run_mapper(job, colmap, c, c["mapper"], rig=rig)

    best, best_stats, models, txt_root = pick_model(job, colmap)
    reset_dir(job.sparse)
    shutil.copytree(best, job.sparse / "0")
    for f in (txt_root / best.name).glob("*.txt"):  # TXT copy for trainers that read text
        shutil.copy2(f, job.sparse / "0" / f.name)
    total = len(list(job.images.rglob("*.jpg")))
    best_stats.update({"total_images": total, "models": models, "chosen_model": best.name,
                       "registered_ratio": best_stats["registered"] / max(total, 1)})
    best_stats.update(extra_stats)
    job.write_json("sfm_stats.json", best_stats)
    log.info("SfM: %d/%d images registered (%.0f%%), parallax ratio %.3f",
             best_stats["registered"], total, 100 * best_stats["registered_ratio"],
             best_stats["parallax_ratio"])


# --------------------------------------------------------------------------- #
# Stage: gate
# --------------------------------------------------------------------------- #
def stage_gate(job: Job, skip: bool = False):
    c = job.cfg["gate"]
    s = job.read_json("sfm_stats.json")
    problems = []
    if s["registered_ratio"] < c["min_registered_ratio"]:
        problems.append(f"only {s['registered_ratio']:.0%} of images registered "
                        f"(need {c['min_registered_ratio']:.0%}). Try more frames, slower "
                        "camera motion, or the exhaustive matcher.")
    if s["parallax_ratio"] < c["min_parallax_ratio"]:
        problems.append(f"camera path is tiny relative to the room (parallax ratio "
                        f"{s['parallax_ratio']:.3f} < {c['min_parallax_ratio']}). The camera "
                        "needs to move through the space, not hover in one spot.")
    if s.get("models", 1) > 1:
        log.warning("Capture split into %d models; only the largest is used", s["models"])
    if problems:
        msg = "Capture quality gate failed:\n  - " + "\n  - ".join(problems)
        if skip:
            log.warning("%s\n(continuing because --skip-gate was given)", msg)
        else:
            raise GateError(msg)
    else:
        log.info("Quality gate passed")


# --------------------------------------------------------------------------- #
# Stage: train / export
# --------------------------------------------------------------------------- #
def template_values(job: Job) -> dict:
    t = job.cfg["train"]
    vals = {"data": job.dir, "train_dir": job.train_dir, "export_dir": job.export_dir,
            "iterations": t["iterations"], "name": job.name,
            "python": sys.executable, "script_dir": Path(__file__).resolve().parent,
            "config_path": job.cfg.get("_config_path", "")}
    vals.update(t.get("vars") or {})
    return vals


def ensure_txt_model(job: Job):
    """Older runs only stored binary files in sparse/0; copy the TXT conversion in."""
    dst = job.sparse / "0"
    if (dst / "images.txt").exists():
        return
    chosen = job.read_json("sfm_stats.json").get("chosen_model", "0")
    src = job.dir / "sparse_txt" / chosen
    if not (src / "images.txt").exists():
        raise PipelineError(f"No TXT model found in {dst} or {src}; re-run with --from-stage sfm")
    for f in src.glob("*.txt"):
        shutil.copy2(f, dst / f.name)
    log.info("Copied TXT model from %s into %s", src, dst)


def stage_train(job: Job):
    t = job.cfg["train"]
    template = (t.get("commands") or {}).get(t["backend"])
    if not template:
        raise PipelineError(f"No train.commands.{t['backend']} template in config")
    ensure_txt_model(job)
    if t["backend"] == "custom":
        job.train_dir.mkdir(parents=True, exist_ok=True)  # keep ckpt.pt so runs can resume
    else:
        reset_dir(job.train_dir)
    cmd = fill_template(template, template_values(job))
    if t["backend"] == "custom" and not job.cfg.get("_config_path"):
        cmd = cmd[:-2]  # no config file: drop the empty --config argument
    t0 = time.time()
    run(cmd, cwd=job.dir, echo=True)
    log.info("Training finished in %.1f min", (time.time() - t0) / 60)


def newest(paths) -> Path | None:
    paths = list(paths)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def write_scene_bundle(job: Job, dest: Path):
    """Everything needed to keep refining a splat later without redoing SfM (e.g. the Fixer
    stage): the camera model without the bulky 2D observations, which frames were used,
    the view layout and the settings. A few MB; the images themselves are re-cut from the
    original video on demand."""
    src = job.sparse / "0"
    if not (src / "images.txt").exists():
        log.warning("No TXT model to bundle; skipping scene bundle")
        return
    sp_out = reset_dir(dest / "sparse" / "0")
    shutil.copy2(src / "cameras.txt", sp_out / "cameras.txt")
    with open(src / "images.txt") as fin, open(sp_out / "images.txt", "w") as fout:
        lines = [l for l in fin if not l.startswith("#")]
        for i in range(0, len(lines), 2):
            if len(lines[i].split()) >= 10:
                fout.write(lines[i].rstrip("\n") + "\n\n")  # pose line, no 2D points
    with open(src / "points3D.txt") as fin, open(sp_out / "points3D.txt", "w") as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 8:
                fout.write(" ".join(p[:8]) + "\n")  # id xyz rgb error, no tracks
    for name in ("intrinsics.json", "selection.json", "video.json", "sfm_stats.json"):
        if (job.dir / name).exists():
            shutil.copy2(job.dir / name, dest / name)
    (dest / "config.yaml").write_text(yaml.safe_dump(
        {k: v for k, v in job.cfg.items() if not k.startswith("_")}, sort_keys=False))
    ts = job.train_dir / "train_stats.json"
    if ts.exists():  # the exact trainer settings the splat was made with (e.g. anti-aliasing)
        try:
            settings = json.loads(ts.read_text()).get("settings") or {}
            (dest / "train_settings.json").write_text(json.dumps(settings, indent=2))
        except (OSError, ValueError) as e:
            log.warning("Could not record training settings in the bundle: %s", e)
    pa = job.train_dir / "pose_adjust.json"
    if pa.exists():  # refined cameras: the Fixer step must render from the same poses
        shutil.copy2(pa, dest / "pose_adjust.json")
    log.info("Scene bundle written to %s", dest)


def stage_export(job: Job):
    backend = job.cfg["train"]["backend"]
    template = (job.cfg["export"].get("commands") or {}).get(backend)
    reset_dir(job.export_dir)
    if template:
        vals = template_values(job)
        cfg_file = newest(job.train_dir.rglob("config.yml"))
        if cfg_file is not None:
            vals["config"] = cfg_file
        run(fill_template(template, vals), cwd=job.dir)
    ply = newest(job.export_dir.rglob("*.ply")) or newest(job.train_dir.rglob("*.ply"))
    if ply is None:
        raise PipelineError("No .ply found after training/export. Check the trainer's save "
                            "options (e.g. gsplat needs --save_ply).")
    job.out.mkdir(parents=True, exist_ok=True)
    dest = job.out / f"{job.name}.ply"
    shutil.copy2(ply, dest)
    stats = job.read_json("sfm_stats.json")
    (job.out / f"{job.name}.json").write_text(json.dumps(
        {"source": str(job.video), "ply": str(dest), "sfm": stats,
         "backend": backend, "iterations": job.cfg["train"]["iterations"]}, indent=2))
    log.info("Splat written to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    try:
        write_scene_bundle(job, job.out / f"{job.name}_scene")
    except Exception as e:  # never lose a finished splat over the bundle
        log.warning("Could not write the scene bundle: %s", e)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_job(job: Job, from_stage: str | None, until_stage: str | None, skip_gate: bool):
    if from_stage:
        job.invalidate_from(from_stage)
    last = STAGES.index(until_stage) if until_stage else len(STAGES) - 1
    funcs = {"extract": stage_extract, "select": stage_select, "slice": stage_slice,
             "sfm": stage_sfm, "gate": lambda j: stage_gate(j, skip_gate),
             "train": stage_train, "export": stage_export}
    for stage in STAGES[: last + 1]:
        if job.done(stage):
            log.info("[%s] %s: already done, skipping", job.name, stage)
            continue
        log.info("[%s] === %s ===", job.name, stage)
        t0 = time.time()
        funcs[stage](job)
        job.mark(stage, {"seconds": round(time.time() - t0, 1)})


def run_pipeline(videos: list[Path], cfg: dict, work: Path, out: Path,
                 from_stage: str | None = None, until_stage: str | None = None,
                 skip_gate: bool = False) -> dict:
    """Process videos and return {video_name: {"status": ..., "ply": ..., "error": ...}}.
    This is the entry point a container / compute-module wrapper should call."""
    results = {}
    for video in videos:
        job = Job(Path(video), Path(work), Path(out), cfg)
        fh = logging.FileHandler(job.dir / "pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(fh)
        res = {"status": "ok", "ply": None, "error": None, "work_dir": str(job.dir)}
        try:
            run_job(job, from_stage, until_stage, skip_gate)
            ply = job.out / f"{job.name}.ply"
            res["ply"] = str(ply) if ply.exists() else None
        except GateError as e:
            log.error("[%s] %s", job.name, e)
            res.update(status="rejected", error=str(e))
        except PipelineError as e:
            log.error("[%s] %s", job.name, e)
            res.update(status="failed", error=str(e))
        finally:
            log.removeHandler(fh)
            fh.close()
        results[Path(video).name] = res
    return results


def collect_inputs(inputs: list[str]) -> list[Path]:
    vids = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            vids += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            vids.append(p)
        else:
            raise SystemExit(f"Input not found: {p}")
    return vids


def main():
    env = os.environ.get
    ap = argparse.ArgumentParser(description="360 video -> Gaussian splat pipeline")
    ap.add_argument("inputs", nargs="+", help="video files and/or folders of videos")
    ap.add_argument("--config", type=Path, default=env("SPLAT_CONFIG"),
                    help="YAML config (defaults used if omitted)")
    ap.add_argument("--work", type=Path, default=Path(env("SPLAT_WORK", "work")),
                    help="working directory")
    ap.add_argument("--out", type=Path, default=Path(env("SPLAT_OUT", "out")),
                    help="where final .ply files go")
    ap.add_argument("--from-stage", choices=STAGES, help="force re-run from this stage")
    ap.add_argument("--until-stage", choices=STAGES, help="stop after this stage")
    ap.add_argument("--skip-gate", action="store_true", help="warn instead of failing the gate")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    cfg = load_config(args.config)
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    log.addHandler(console)

    try:
        results = run_pipeline(collect_inputs(args.inputs), cfg, args.work, args.out,
                               args.from_stage, args.until_stage, args.skip_gate)
    except KeyboardInterrupt:
        log.error("Interrupted; re-run the same command to resume")
        sys.exit(130)

    log.info("Summary:")
    for name, r in results.items():
        log.info("  %-40s %s%s", name, r["status"], f"  -> {r['ply']}" if r["ply"] else "")
    sys.exit(0 if all(r["status"] == "ok" for r in results.values()) else 1)


if __name__ == "__main__":
    main()
