#!/usr/bin/env python3
"""
mapanything_init.py - metric camera poses from MapAnything, used to seed COLMAP.

Used by splat_pipeline.py when sfm.mapper is "mapanything". It runs as a separate process
so MapAnything's dependencies and GPU memory never touch the main pipeline (and it can run
under a different Python via sfm.mapanything.python).

What it does
  1. Picks a few sliced views per 360 frame (default: the four horizontal ones at 90 deg).
  2. Runs MapAnything (Apache-2.0 weights by default) on sliding windows of frames, giving
     it our exact pinhole intrinsics, and gets metric camera-to-world poses.
  3. Every view of one 360 frame shares a single physical camera position and known relative
     rotations, so it fuses each frame's view poses into one "rig" pose (and reports how much
     the views disagreed - a quality signal for MapAnything itself).
  4. Chains the windows into one metric world using the overlapping frames.
  5. Derives poses for ALL views of every frame and writes them to poses.json.

COLMAP then triangulates points with these poses fixed and bundle-adjusts (see
splat_pipeline.refine_with_colmap), so final accuracy comes from COLMAP's feature matches
while scale and speed come from MapAnything.

Usage (normally called by splat_pipeline.py):
    python mapanything_init.py --data work/patio --out work/patio/ma_init --settings '{...}'
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("mapanything")

DEFAULTS = {
    "model": "facebook/map-anything-apache",  # Apache-2.0 weights (commercial use OK)
    "views": "auto",          # list of view indices, or "auto" = horizontal views ~90 deg apart
    "window_frames": 16,      # 360 frames per MapAnything call (x views each = images per call)
    "overlap_frames": 4,      # frames shared by consecutive windows (used to chain them)
    "window_scale": "estimate",  # "estimate" per-window scale from overlaps, or "fixed" (=1)
    "minibatch_size": 1,      # MapAnything memory-efficient inference; 1 = lowest VRAM
    "amp_dtype": "bf16",      # falls back to fp16 on GPUs without bf16 (e.g. RTX 20xx)
}


# --------------------------------------------------------------------------- #
# Geometry helpers (pure numpy; imported and unit-tested by the pipeline)
# --------------------------------------------------------------------------- #
def rig_from_cam(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation taking camera rays to the level 360 rig frame. Identical to
    train_gsplat.rig_from_cam and splat_pipeline.build_view_maps (x right, y down, z fwd)."""
    p, t = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    return Ry @ Rx


def project_rotation(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (chordal mean when M is a sum of rotations)."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def parse_name(name: str):
    """'v03/f00012.jpg' (or legacy 'f00012_v03.jpg') -> ('f00012', 3)."""
    try:
        import naming
    except ImportError:  # run from another folder
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import naming
    parsed = naming.parse(name)
    return parsed if parsed else (None, None)


def auto_views(views: list) -> list[int]:
    """Horizontal views roughly 90 deg apart: good coverage with few images."""
    horiz = [(i, v[0] % 360) for i, v in enumerate(views) if abs(v[1]) < 1e-6]
    picked = []
    for target in (0, 90, 180, 270):
        best = min(horiz, key=lambda iv: min(abs(iv[1] - target), 360 - abs(iv[1] - target)),
                   default=None)
        if best and best[0] not in picked:
            picked.append(best[0])
    return picked or list(range(min(4, len(views))))


def fuse_rig(cam_poses: dict, views: list):
    """cam_poses: {view_idx: 4x4 cam-to-world}. Returns (R_world_from_rig, centre, spread_deg)."""
    Rs, cs = [], []
    for vi, T in cam_poses.items():
        R_wc = T[:3, :3]
        Rs.append(R_wc @ rig_from_cam(*views[vi][:2]).T)
        cs.append(T[:3, 3])
    R = project_rotation(sum(Rs))
    spread = max(rot_angle_deg(R.T @ r) for r in Rs) if len(Rs) > 1 else 0.0
    return R, np.mean(cs, axis=0), spread


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    size = max(2, size)
    step = max(1, size - max(0, min(overlap, size - 1)))
    out, start = [], 0
    while True:
        end = min(n, start + size)
        out.append((start, end))
        if end >= n:
            return out
        start += step


def align_window(acc: dict, win: dict, estimate_scale: bool, min_spread: float = 0.05):
    """Similarity mapping window coords -> accumulated coords from shared frames.
    Rotation comes from the rig orientations (robust even if the drone barely moved),
    scale from camera-centre spread, translation from centroids."""
    shared = [f for f in win if f in acc]
    if not shared:
        raise RuntimeError("window has no frames in common with the previous ones")
    R = project_rotation(sum(acc[f][0] @ win[f][0].T for f in shared))
    ca = np.array([acc[f][1] for f in shared])
    cw = np.array([win[f][1] for f in shared]) @ R.T
    ma, mw = ca.mean(0), cw.mean(0)
    sa = np.linalg.norm(ca - ma, axis=1).sum()
    sw = np.linalg.norm(cw - mw, axis=1).sum()
    s = float(sa / sw) if (estimate_scale and len(shared) >= 2 and sw > min_spread * len(shared)
                           and sa > 0) else 1.0
    t = ma - s * mw
    resid = float(np.sqrt(np.mean(np.sum((s * cw + t - ca) ** 2, axis=1))))
    return R, s, t, resid, len(shared)


def chain_windows(window_rigs: list[dict], estimate_scale: bool):
    """window_rigs: list of {frame: (R_world_from_rig, centre)} in each window's own frame.
    Returns ({frame: (R, centre)} in the first window's world, per-window report)."""
    acc = dict(window_rigs[0])
    report = [{"window": 0, "frames": len(window_rigs[0]), "scale": 1.0, "residual_m": 0.0}]
    for k, win in enumerate(window_rigs[1:], start=1):
        R, s, t, resid, n_shared = align_window(acc, win, estimate_scale)
        for f, (Rf, cf) in win.items():
            if f not in acc:
                acc[f] = (R @ Rf, s * (R @ cf) + t)
        report.append({"window": k, "frames": len(win), "shared": n_shared,
                       "scale": round(s, 4), "residual_m": round(resid, 4)})
    return acc, report


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if (with_scale and var > 1e-12) else 1.0
    t = md - s * R @ ms
    return s, R, t


def rotmat_to_qvec(R: np.ndarray) -> list[float]:
    """Rotation matrix -> COLMAP quaternion [w, x, y, z] (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q) / np.linalg.norm(q)
    return [float(v) for v in (-q if q[0] < 0 else q)]


def all_view_poses(rigs: dict, frames_to_images: dict, views: list) -> dict:
    """COLMAP cam_from_world (R, t) for every image of every posed frame."""
    out = {}
    for frame, (R_wr, c) in rigs.items():
        for name, vi in frames_to_images.get(frame, []):
            R_wc = R_wr @ rig_from_cam(*views[vi][:2])
            R = R_wc.T
            out[name] = (R, -R @ c)
    return out


# --------------------------------------------------------------------------- #
# MapAnything inference
# --------------------------------------------------------------------------- #
def run(data_dir: Path, out_dir: Path, s: dict):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import cv2
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    intr = json.loads((data_dir / "intrinsics.json").read_text())
    views = intr["views"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
                 np.float32)
    use_views = auto_views(views) if s["views"] == "auto" else [int(v) for v in s["views"]]

    frames_to_images: dict[str, list] = {}
    img_root = data_dir / "images"
    for p in sorted(img_root.rglob("*.jpg")):
        rel = p.relative_to(img_root).as_posix()  # COLMAP image name, e.g. v03/f00012.jpg
        frame, vi = parse_name(rel)
        if frame is not None and vi < len(views):
            frames_to_images.setdefault(frame, []).append((rel, vi))
    frame_list = sorted(frames_to_images)
    if len(frame_list) < 2:
        raise RuntimeError("Need at least 2 frames of sliced views in images/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No CUDA GPU: MapAnything will be very slow on CPU")
    amp = s["amp_dtype"]
    if device == "cuda" and amp == "bf16" and not torch.cuda.is_bf16_supported():
        amp = "fp16"
    log.info("Loading %s on %s (amp %s); views %s per frame, %d frames",
             s["model"], device, amp, use_views, len(frame_list))
    model = MapAnything.from_pretrained(s["model"]).to(device).eval()

    wins = windows(len(frame_list), int(s["window_frames"]), int(s["overlap_frames"]))
    window_rigs, spreads = [], []
    t0 = time.time()
    for wi, (a, b) in enumerate(wins):
        batch, keys = [], []
        for frame in frame_list[a:b]:
            for name, vi in frames_to_images[frame]:
                if vi not in use_views:
                    continue
                img = cv2.cvtColor(cv2.imread(str(data_dir / "images" / name)), cv2.COLOR_BGR2RGB)
                batch.append({"img": torch.from_numpy(img), "intrinsics": torch.from_numpy(K)})
                keys.append((frame, vi))
        if len(batch) < 2:
            continue
        processed = preprocess_inputs(batch, verbose=False)
        with torch.no_grad():
            preds = model.infer(processed, memory_efficient_inference=True,
                                minibatch_size=int(s["minibatch_size"]), use_amp=device == "cuda",
                                amp_dtype=amp, apply_mask=True, mask_edges=True,
                                ignore_calibration_inputs=False, ignore_depth_inputs=True,
                                ignore_pose_inputs=True)
        per_frame: dict[str, dict] = {}
        for (frame, vi), pred in zip(keys, preds):
            T = pred["camera_poses"][0].float().cpu().numpy().astype(np.float64)
            per_frame.setdefault(frame, {})[vi] = T
        rigs = {}
        for frame, cams in per_frame.items():
            R, c, spread = fuse_rig(cams, views)
            rigs[frame] = (R, c)
            spreads.append(spread)
        window_rigs.append(rigs)
        log.info("MapAnything window %d/%d: frames %d-%d, %d images (%.0f s so far)",
                 wi + 1, len(wins), a, b - 1, len(batch), time.time() - t0)
        del preds, processed
        if device == "cuda":
            torch.cuda.empty_cache()

    if not window_rigs:
        raise RuntimeError("MapAnything produced no poses")
    rigs, chain_report = chain_windows(window_rigs, s["window_scale"] == "estimate")
    poses = all_view_poses(rigs, frames_to_images, views)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "poses.json").write_text(json.dumps(
        {name: {"qvec": rotmat_to_qvec(R), "tvec": list(map(float, t))}
         for name, (R, t) in poses.items()}))
    (out_dir / "rigs.json").write_text(json.dumps(
        {f: {"R": R.tolist(), "center": c.tolist()} for f, (R, c) in rigs.items()}))
    centres = np.array([c for _, c in rigs.values()])
    report = {
        "model": s["model"], "views_used": use_views, "frames_posed": len(rigs),
        "frames_total": len(frame_list), "images_posed": len(poses),
        "windows": chain_report,
        "view_agreement_deg": {"median": round(float(np.median(spreads)), 3),
                               "p95": round(float(np.percentile(spreads, 95)), 3)},
        "path_extent_m": round(float(np.linalg.norm(centres.max(0) - centres.min(0))), 3),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    log.info("MapAnything: %d/%d frames posed, views agree within %.2f deg (median), "
             "flight path spans %.2f m, %.0f s", report["frames_posed"], report["frames_total"],
             report["view_agreement_deg"]["median"], report["path_extent_m"], report["seconds"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", default="{}", help="JSON overrides of the defaults")
    args = ap.parse_args()
    s = dict(DEFAULTS, **json.loads(args.settings))
    try:
        run(args.data.resolve(), args.out.resolve(), s)
    except Exception as e:
        log.error("%s: %s", type(e).__name__, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
