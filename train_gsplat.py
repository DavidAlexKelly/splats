#!/usr/bin/env python3
"""
train_gsplat.py - Gaussian splat trainer for COLMAP datasets, built on gsplat.

Reads a COLMAP dataset (images/ + sparse/0 with TXT model files, optional
masks/), trains 3D Gaussians with gsplat's CUDA rasteriser and writes a
standard 3DGS .ply (opens in SuperSplat, Brush, PlayCanvas and most viewers).

Usage:
    python train_gsplat.py --check                                   # test torch + CUDA + gsplat
    python train_gsplat.py --data work/patio --out work/patio/train
    python train_gsplat.py --data work/patio --out work/patio/train --config config.yaml

Normally called by splat_pipeline.py (train.backend: custom). Settings come
from the train.custom block of config.yaml; command-line flags override them.

Resumes automatically from <out>/ckpt.pt if training was interrupted.
Delete the output folder to force a fresh run.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import appearance  # noqa: E402
import scaffold as scaffold_mod  # noqa: E402
import depth_prior  # noqa: E402
import naming  # noqa: E402
import scene_fill  # noqa: E402

log = logging.getLogger("train")

DEFAULTS = {
    "iterations": 30000,
    "strategy": "mcmc",          # "mcmc" (fixed Gaussian budget, predictable VRAM) or "default"
    "cap_max": 1_000_000,        # MCMC: max number of Gaussians. ~1M fits an 8 GB GPU comfortably
    "sh_degree": 2,              # view-dependent colour detail (0-3). 2: less sparkle from
                                 # angles the drone never had, matches viewers' top setting
    "sh_degree_interval": 1000,  # add one SH band every N steps
    "downscale": 1,              # train on images downscaled by this factor
    "ssim_lambda": 0.2,
    "opacity_reg": 0.01,         # MCMC regularisers
    "scale_reg": 0.01,
    "random_background": False,
    "packed": True,              # lower-memory rasterisation
    "cache_images": False,       # keep decoded images in RAM (~7.7 MB each at 1600px)
    "loader_threads": 4,
    "log_every": 100,
    "save_every": 5000,
    "near_plane": 0.01,
    "seed": 42,
    "align": True,               # rotate the scene upright using the known 360 view directions
    "up_axis": "+y",             # "+y" SuperSplat/PlayCanvas/three.js, "+z" Blender/Unreal, "-y" COLMAP
    "world_scale": 1.0,          # multiply scene size (COLMAP units are arbitrary)
    "exposure": True,            # learn a colour/exposure correction per photo
    "exposure_lr": 1e-3,
    "exposure_reg": 1e-2,        # keeps corrections small so true colours stay in the splat
    "bilagrid": False,           # per-photo bilateral grid. Off: it lets floaters near the camera
                                 # hide behind per-photo corrections (dark blobs in viewers)
    "bilagrid_xy": 16,           # grid cells across the image
    "bilagrid_luma": 8,          # grid cells along brightness
    "bilagrid_lr": 2e-3,
    "bilagrid_tv": 10.0,         # smoothness: the grids may only make gentle, broad changes
    "antialiased": True,         # Mip-Splatting style anti-aliasing: less shimmer off the path
    # Solid-or-gone: late in training, push every Gaussian to clearly opaque or invisible.
    # Thin see-through layers are what flickers ("pops") as a viewer re-sorts them.
    "opacity_entropy": 0.002,
    "opacity_entropy_start": 0.6,  # fraction of training before it starts (ramps in over 10%)
    # ... and when it stops. 0 = when MCMC stops relocating (85%): after that nothing recycles
    # the Gaussians it pushes to ~0 (on patio 61% ended below 0.005 opacity yet still held
    # 1.4 dB together). 1 = to the end, as before.
    "opacity_entropy_end": 0.0,
    # Camera refinement: one small pose correction per 360 frame (shared by its cube faces, so
    # the rig stays rigid). Fixes slight pose errors that show up as doubled edges and ghosts.
    "pose_opt": False,           # off by default: on a rig solve it moved cameras ~0.03%
    "pose_lr": 1e-5,
    "pose_reg": 1e-6,
    "pose_opt_start": 1000,
    # Scene scaffold (scaffold.py, scaffold_agent.py): the big surfaces (floor, walls, glass)
    # with a rule each. solid: exact depth + empty space in front + flat splats; glass: the
    # same, placed from its outline (depth sees through glass); soft (grass): only the space
    # well in front must stay empty. Claude decides the rules when ANTHROPIC_API_KEY is set,
    # otherwise automatic rules.
    "scaffold": True,
    "scaffold_mode": "claude",     # claude (falls back to auto without a key) | auto
    "scaffold_decision": "",       # reuse a saved scaffold_decision.json (edit it to override)
    "scaffold_redo": False,        # ignore this run's own saved decision (after a restart)
    "scaffold_model": "claude-opus-5-5",
    "scaffold_max_cost": 3.0,      # USD, estimated from tokens; stops the agent when reached
    "scaffold_max_turns": 40,
    "scaffold_thinking": 0,        # extended-thinking budget per turn (0 = model default)
    "scaffold_price_in": 5.0,      # USD per million input tokens (for the estimate only)
    "scaffold_price_out": 25.0,    # USD per million output tokens
    "scaffold_prompt": "",         # custom instructions file (default: scaffold_prompt.md)
    "scaffold_tol": 0.01,          # plane thickness, x typical camera-to-surface distance
    "scaffold_cell": 0.015,        # outline cell size, x the same distance
    "scaffold_candidates": 40,     # how many candidate surfaces to look for
    "scaffold_cand_min_frac": 0.002,  # a candidate needs >= this share of the SfM points
    "scaffold_ransac_iters": 300,
    "scaffold_max_tilt_deg": 10,   # within this of level = floor/ceiling, of upright = wall
    "scaffold_check_images": 48,   # photos used to check candidates against depth
    "scaffold_check_step": 8,      # ... sampled every 8 pixels
    "scaffold_min_views": 3,       # photos that must confirm a surface (automatic rules)
    "scaffold_min_area": 0.005,    # x (typical distance)^2 (automatic rules)
    "scaffold_min_concentration": 0.5,  # share of nearby points ON the plane (bush ~0.25)
    "scaffold_photo_min": 0.5,     # share of photo depth agreeing with the plane
    "scaffold_res_step": 4,        # surface depth computed at 1/4 resolution in training
    "scaffold_agree": 0.05,        # depth estimate within 5% of a surface = it's visible
    "scaffold_conf": 1.0,          # depth weight on visible solid/glass pixels
    "scaffold_weight": 0.05,       # empty-space penalty weight (constant through training)
    "scaffold_margin": 0.03,       # solid/glass: >3% in front of the surface = floater
    "scaffold_soft_margin": 0.15,  # soft: >15% in front = floater (grass keeps its height)
    "scaffold_flat_weight": 0.02,  # flat splats on solid/glass surfaces (0 = off)
    "scaffold_flat_start": 2000,
    "scaffold_flat_every": 100,    # re-find which splats lie on a surface every N steps
    # A short video along the flight path, shifted slightly sideways: shows popping/floaters
    # that a still image or the held-out score can't.
    "flythrough": True,
    "flythrough_offset": 0.03,     # sideways shift, x scene scale
    "test_every": 20,            # hold out every Nth image for an honest score (0 = off)
    "eval_every": 5000,
    # Monocular depth prior (see depth_prior.py)
    "depth_prior": False,
    "depth_backend": "auto",     # "auto" (DA3 mono if installed, else DA2 Small), "da3", "da2"
    "depth_model": "",           # empty = backend default (both Apache-2.0)
    "depth_process_res": 756,    # DA3 inference resolution (multiple of 14)
    "depth_batch": 8,
    "depth_mode": "selective",   # "selective": weak-geometry areas only; "uniform": everywhere
    "depth_min_weight": 0.15,    # weight where SfM points are dense
    "depth_weight": 0.1,         # loss weight at the start of training
    "depth_end_ratio": 0.1,      # weight decays linearly to depth_weight * this by the end
    "depth_grad_weight": 1.0,    # depth-gradient (shape) term relative to the disparity term
    "depth_min_points": 30,      # SfM points an image needs for its prior to be aligned
    "depth_max_rel_error": 0.35, # reject priors whose alignment error exceeds this
    # Filling gaps from real data (see scene_fill.py)
    "depth_seed": True,          # extra starting Gaussians from the depth prior in thin areas
    "depth_seed_max_points": 300_000,
    "depth_seed_min_weight": 0.5,  # only where the prior's coverage weight says SfM is sparse
    "depth_seed_stride": 8,      # sample every Nth pixel
    "depth_seed_voxel": 0.004,   # merge seeds closer than this x scene scale
    "sky_shell": True,           # sphere of Gaussians far away for sky / distant background
    "sky_points": 100_000,
    "sky_radius_mult": 3.0,      # shell radius = this x the scene's extent
    "prune": True,               # floater cleanup before export (checkpoint keeps everything)
    "prune_min_opacity": 0.005,  # MCMC keeps many faint Gaussians that matter together
    "prune_max_psnr_drop": 0.05, # undo/lighten the cleanup if held-out PSNR drops more
    "prune_near_camera": 0.02,   # remove Gaussians closer than this x scene scale to a camera
    "prune_max_scale": 1.0,      # remove blobs larger than this x scene scale (not the sky)
    "prune_outlier_std": 3.0,    # remove isolated specks (0 = off)
}

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def resolve_cap_max(value, total_mem_bytes: int | None) -> int:
    """cap_max "auto": the Gaussian budget follows GPU memory (each Gaussian with its optimiser
    state and gradients costs ~1 KB, plus rendering buffers). 3M is what a 16 GB T4 handles."""
    if str(value).lower() != "auto":
        return int(value)
    gb = (total_mem_bytes or 0) / 1e9
    if gb < 20:
        return 3_000_000
    if gb < 28:
        return 4_000_000   # 24 GB: RTX 4090 / 3090, L4
    if gb < 40:
        return 5_000_000   # 32 GB: RTX 5090
    return 6_000_000       # 48 GB+: L40S, A100, H100, RTX PRO 6000


def load_settings(config_path: str | None, overrides: dict) -> dict:
    s = dict(DEFAULTS)
    if config_path:
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        train = cfg.get("train", {}) or {}
        if "iterations" in train:
            s["iterations"] = train["iterations"]
        s.update(train.get("custom", {}) or {})
        try:
            s["_views"] = naming.resolve_views(cfg.get("slice", {}) or {})
        except ValueError:
            pass
    s.update({k: v for k, v in overrides.items() if v is not None})
    return s


# --------------------------------------------------------------------------- #
# COLMAP text model
# --------------------------------------------------------------------------- #
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras_txt(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        cid, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
        params = list(map(float, p[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Camera model {model} is not supported; expected undistorted "
                             "PINHOLE images (splat_pipeline.py produces these).")
        cams[cid] = {"w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cams


def read_images_txt(path: Path) -> list[dict]:
    lines = [l for l in path.read_text().splitlines(keepends=True) if not l.startswith("#")]
    out = []
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        R = qvec2rotmat(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        obs = lines[i + 1].split() if i + 1 < len(lines) else []
        arr = np.array(obs, dtype=np.float64).reshape(-1, 3) if obs else np.zeros((0, 3))
        arr = arr[arr[:, 2] >= 0]  # keep 2D points that triangulated to a 3D point
        out.append({"name": p[9], "camera_id": int(p[8]), "R": R, "t": t,
                    "obs_xy": arr[:, :2], "obs_id": arr[:, 2].astype(np.int64)})
    return out


def read_points_txt(path: Path, with_ids: bool = False):
    xyz, rgb, ids = [], [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        ids.append(int(p[0]))
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
    out = (np.array(xyz, np.float32).reshape(-1, 3), np.array(rgb, np.float32).reshape(-1, 3) / 255.0)
    return out + (np.array(ids, np.int64),) if with_ids else out


# --------------------------------------------------------------------------- #
# Gravity alignment
# --------------------------------------------------------------------------- #
rig_from_cam = naming.rig_from_cam  # camera rays -> level 360 frame (shared with the slicer)


TARGET_AXES = {  # (up, back) in the output world; forward = -back
    "+y": (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    "-y": (np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    "+z": (np.array([0, 0, 1.0]), np.array([0, -1.0, 0])),
}


def compute_alignment(images: list[dict], views: list, up_axis: str, scale: float = 1.0):
    """Returns (4x4 transform world->aligned, stats). Uses each image's known view
    direction inside the level 360 frame to recover gravity and the initial heading."""
    up_rig, fwd_rig = np.array([0, -1.0, 0]), np.array([0, 0, 1.0])
    ups, fwds, centers = [], [], []
    first_frame = None
    for im in sorted(images, key=lambda d: d["name"]):
        parsed = naming.parse(im["name"])
        if not parsed or parsed[1] >= len(views):
            continue
        frame, vi = parsed
        R_rc = rig_from_cam(*views[vi][:2])
        world_from_cam = im["R"].T
        ups.append(world_from_cam @ R_rc.T @ up_rig)
        if first_frame is None:
            first_frame = frame
        if frame == first_frame:
            fwds.append(world_from_cam @ R_rc.T @ fwd_rig)
        centers.append(-im["R"].T @ im["t"])
    if len(ups) < 10:
        return None, {"error": "too few images with view names (vNN/...) to align"}
    ups = np.array(ups)
    u = ups.mean(0)
    u /= np.linalg.norm(u)
    dev = np.degrees(np.arccos(np.clip(ups @ u, -1, 1)))
    f = np.mean(fwds, axis=0)
    f = f - (f @ u) * u
    if np.linalg.norm(f) < 1e-6:  # degenerate: pick any horizontal direction
        f = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    f /= np.linalg.norm(f)
    src_y, src_z = u, -f
    src = np.stack([np.cross(src_y, src_z), src_y, src_z], axis=1)
    t_up, t_back = TARGET_AXES[up_axis]
    tgt = np.stack([np.cross(t_up, t_back), t_up, t_back], axis=1)
    A = tgt @ src.T
    c = np.mean(centers, axis=0)
    M = np.eye(4)
    M[:3, :3] = scale * A
    M[:3, 3] = -scale * A @ c
    stats = {"images_used": int(len(ups)), "up_deviation_median_deg": float(np.median(dev)),
             "up_deviation_p95_deg": float(np.percentile(dev, 95)), "up_axis": up_axis,
             "world_scale": scale}
    return M, stats


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ColmapDataset:
    def __init__(self, data_dir: Path, downscale: float = 1, cache: bool = False,
                 align: bool = False, views: list | None = None, up_axis: str = "+y",
                 world_scale: float = 1.0):
        sparse = data_dir / "sparse" / "0"
        self.images_dir = data_dir / "images"
        self.masks_dir = data_dir / "masks"
        for f in ("cameras.txt", "images.txt", "points3D.txt"):
            if not (sparse / f).exists():
                raise FileNotFoundError(
                    f"{sparse / f} not found. Convert the model to TXT first:\n"
                    f"  colmap model_converter --input_path {sparse} --output_path {sparse} "
                    "--output_type TXT")
        cams = read_cameras_txt(sparse / "cameras.txt")
        images = sorted(read_images_txt(sparse / "images.txt"), key=lambda d: d["name"])
        self.points, self.colors, point_ids = read_points_txt(sparse / "points3D.txt", with_ids=True)
        id_to_idx = {int(pid): k for k, pid in enumerate(point_ids)}
        if len(images) == 0 or len(self.points) == 0:
            raise ValueError("COLMAP model has no images or no points")

        self.alignment, self.align_stats = None, None
        self.align_up = up_axis
        if align:
            M, st = (compute_alignment(images, views, up_axis, world_scale) if views
                     else (None, {"error": "no view list (intrinsics.json or config slice.views)"}))
            self.align_stats = st
            if M is not None:
                self.alignment = M
                Minv = np.linalg.inv(M)
                self.points = (self.points @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
                for im in images:  # w2c' = w2c @ M^-1
                    w2c = np.eye(4)
                    w2c[:3, :3], w2c[:3, 3] = im["R"], im["t"]
                    w2c = w2c @ Minv
                    w2c[:3, :] *= world_scale  # keep R orthonormal; camera space scales too
                    im["R"], im["t"] = w2c[:3, :3], w2c[:3, 3]

        self.frames = []
        for im in images:
            c = cams[im["camera_id"]]
            W, H = max(1, round(c["w"] / downscale)), max(1, round(c["h"] / downscale))
            sx, sy = W / c["w"], H / c["h"]
            K = np.array([[c["fx"] * sx, 0, c["cx"] * sx],
                          [0, c["fy"] * sy, c["cy"] * sy],
                          [0, 0, 1]], np.float32)
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = im["R"]
            viewmat[:3, 3] = im["t"]
            mask = self.masks_dir / f"{im['name']}.png"
            keep = [k for k, pid in enumerate(im["obs_id"]) if int(pid) in id_to_idx]
            self.frames.append({
                "name": im["name"], "path": self.images_dir / im["name"],
                "mask": mask if mask.exists() else None,
                "K": K, "viewmat": viewmat, "W": W, "H": H,
                "src_w": c["w"], "src_h": c["h"],
                "center": -im["R"].T @ im["t"],
                "obs_uv": im["obs_xy"][keep],
                "obs_pidx": np.array([id_to_idx[int(im["obs_id"][k])] for k in keep], np.int64),
            })
        if sum(len(f["obs_pidx"]) for f in self.frames) == 0:
            # a scene bundle keeps poses but not which points each photo saw: estimate it
            pseudo_observations(self.frames, self.points)
        centers = np.array([f["center"] for f in self.frames])
        radius = float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
        self.scene_scale = radius * 1.1 if radius > 0 else 1.0
        self.cache = {} if cache else None
        self.prior_dir = None  # set by depth_prior.prepare()
        self.scaffold = None   # set by train() when scaffold planes were found
        self.scaffold_conf = 1.0

    def load_prior_raw(self, i: int):
        return depth_prior.load(self.prior_dir, self.frames[i]["name"])

    def load_prior_full(self, i: int):
        """(depth, weight, limit or None, surface ids or None) or None."""
        p = depth_prior.load(self.prior_dir, self.frames[i]["name"])
        if p is None:
            return None
        if self.scaffold is None:
            return (p[0], p[1], None, None)
        return self.scaffold.merge_prior(self.frames[i], p[0], p[1], self.scaffold_conf)

    def load_prior(self, i: int):
        p = self.load_prior_full(i)
        return None if p is None else (p[0], p[1])

    def __len__(self):
        return len(self.frames)

    def load(self, i: int):
        f = self.frames[i]
        if self.cache is not None and i in self.cache:
            img, mask = self.cache[i]
        else:
            img = cv2.imread(str(f["path"]), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Could not read {f['path']}")
            if img.shape[1] != f["W"] or img.shape[0] != f["H"]:
                img = cv2.resize(img, (f["W"], f["H"]), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = None
            if f["mask"] is not None:
                m = cv2.imread(str(f["mask"]), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[1] != f["W"] or m.shape[0] != f["H"]:
                        m = cv2.resize(m, (f["W"], f["H"]), interpolation=cv2.INTER_NEAREST)
                    mask = m > 127
            if self.cache is not None:
                self.cache[i] = (img, mask)
        return img, mask


def pseudo_observations(frames: list, points: np.ndarray, cell: int = 16):
    """Which SfM points each photo (probably) saw, from projection: per 16-px cell, the
    points within 3% of the nearest one. Used when the 2D observations were stripped."""
    P = np.asarray(points, np.float64)
    t0 = time.time()
    for f in frames:
        K, V = f["K"].astype(np.float64), f["viewmat"].astype(np.float64)
        Xc = P @ V[:3, :3].T + V[:3, 3]
        z = Xc[:, 2]
        front = z > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            x = K[0, 0] * Xc[:, 0] / z + K[0, 2]
            y = K[1, 1] * Xc[:, 1] / z + K[1, 2]
        ok = front & (x >= 0) & (x < f["W"]) & (y >= 0) & (y < f["H"])
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            continue
        cx, cy = (x[idx] // cell).astype(np.int64), (y[idx] // cell).astype(np.int64)
        key = cy * (f["W"] // cell + 1) + cx
        zmin = np.full(key.max() + 1, np.inf)
        np.minimum.at(zmin, key, z[idx])
        vis = z[idx] <= zmin[key] * 1.03
        keep = idx[vis]
        sx, sy = f["src_w"] / f["W"], f["src_h"] / f["H"]  # obs are at source resolution
        f["obs_uv"] = np.stack([x[keep] * sx, y[keep] * sy], 1)
        f["obs_pidx"] = keep.astype(np.int64)
    log.info("Estimated which points each photo saw (%d photos, %.0f s)", len(frames),
             time.time() - t0)


class Prefetcher:
    """Loads images on background threads (OpenCV releases the GIL) so the GPU never waits."""

    def __init__(self, ds: ColmapDataset, indices: list[int], threads: int = 4, depth: int = 16,
                 seed: int = 0):
        self.ds = ds
        self.indices = list(indices)
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.order: list[int] = []
        self.stop = False
        self.workers = [threading.Thread(target=self._work, daemon=True) for _ in range(threads)]
        for w in self.workers:
            w.start()

    def _next_index(self) -> int:
        with self.lock:
            if not self.order:
                self.order = list(self.indices)
                self.rng.shuffle(self.order)
            return self.order.pop()

    def _work(self):
        while not self.stop:
            i = self._next_index()
            try:
                item = (i, *self.ds.load(i), self.ds.load_prior_full(i))
            except Exception as e:  # surface loader errors in the main thread
                item = e
            while not self.stop:
                try:
                    self.q.put(item, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self):
        item = self.q.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.stop = True


# --------------------------------------------------------------------------- #
# PLY export (standard 3DGS layout)
# --------------------------------------------------------------------------- #
def write_ply(path: Path, means, sh0, shN, opacities, scales, quats):
    """means [N,3]; sh0 [N,1,3]; shN [N,K,3]; opacities [N] (logit); scales [N,3] (log);
    quats [N,4] (wxyz). All numpy float arrays."""
    n = means.shape[0]
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-12)
    f_dc = sh0.reshape(n, 3)
    f_rest = np.transpose(shN, (0, 2, 1)).reshape(n, -1)  # channel-major, as 3DGS expects
    data = np.concatenate([means, np.zeros((n, 3)), f_dc, f_rest, opacities.reshape(n, 1),
                           scales, quats], axis=1).astype(np.float32)
    keep = np.isfinite(data).all(axis=1)
    data = data[keep]
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {data.shape[0]}\n"
              + "".join(f"property float {nm}\n" for nm in names)
              + "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())
    return int(data.shape[0]), int(n - data.shape[0])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def knn_scale(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(points, k=4)
    return np.sqrt((d[:, 1:] ** 2).mean(axis=1)).clip(1e-7)


def make_ssim(torch, F, device):
    size, sigma = 11, 1.5
    g = torch.exp(-((torch.arange(size, dtype=torch.float32) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous().to(device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def ssim(x, y):  # [B,3,H,W] in [0,1]
        pad = size // 2
        mu_x = F.conv2d(x, window, padding=pad, groups=3)
        mu_y = F.conv2d(y, window, padding=pad, groups=3)
        sxx = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x ** 2
        syy = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y ** 2
        sxy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_x * mu_y
        num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2)
        return (num / den).mean()

    return ssim


def frame_view(name: str):
    """'v03/f00012.jpg' (or legacy 'f00012_v03.jpg') -> (12, 3); None otherwise."""
    parsed = naming.parse(name)
    return (naming.frame_number(parsed[0]), parsed[1]) if parsed else None


def split_train_test(ds: "ColmapDataset", test_every: int):
    """Every Nth image (sorted by name) is held out for an honest quality score. Each
    held-out image borrows the exposure of the nearest training frame with the same view."""
    n = len(ds)
    test = [i for i in range(n) if test_every > 0 and i % test_every == 0]
    test_set = set(test)
    train = [i for i in range(n) if i not in test_set]
    by_view: dict[int, list] = {}
    for i in train:
        fv = frame_view(ds.frames[i]["name"])
        if fv:
            by_view.setdefault(fv[1], []).append((fv[0], i))
    expo_src = {}
    for i in test:
        fv = frame_view(ds.frames[i]["name"])
        cands = by_view.get(fv[1], []) if fv else []
        expo_src[i] = min(cands, key=lambda c: abs(c[0] - fv[0]))[1] if cands else None
    return train, test, expo_src


def depth_loss(rend_depth, prior_depth, weight, valid, grad_weight: float):
    """Weighted disparity L1 plus depth-gradient alignment, both normalised by the median
    prior disparity so the loss is independent of scene scale. Gradients of disparity
    capture surface shape (flat walls, floor) even where absolute alignment is imperfect."""
    import torch
    dr = 1.0 / rend_depth.clamp(min=1e-4)
    dp = torch.where(valid, 1.0 / prior_depth.clamp(min=1e-4), torch.zeros_like(prior_depth))
    vf = valid.float()
    if vf.sum() < 64:
        return rend_depth.sum() * 0.0
    norm = dp[valid].median().clamp(min=1e-6)
    w = weight * vf
    l_abs = (w * (dr - dp).abs()).sum() / w.sum().clamp(min=1) / norm
    vx = vf[:, 1:] * vf[:, :-1]
    vy = vf[1:, :] * vf[:-1, :]
    wx = 0.5 * (weight[:, 1:] + weight[:, :-1]) * vx
    wy = 0.5 * (weight[1:, :] + weight[:-1, :]) * vy
    gx = ((dr[:, 1:] - dr[:, :-1]) - (dp[:, 1:] - dp[:, :-1])).abs()
    gy = ((dr[1:, :] - dr[:-1, :]) - (dp[1:, :] - dp[:-1, :])).abs()
    l_grad = ((wx * gx).sum() + (wy * gy).sum()) / (wx.sum() + wy.sum()).clamp(min=1) / norm
    return l_abs + grad_weight * l_grad


def scaffold_up(ds) -> np.ndarray:
    return (TARGET_AXES[ds.align_up][0].astype(np.float64) if ds.alignment is not None
            else np.array([0, -1.0, 0]))  # unaligned: COLMAP's first camera, y down


def build_scaffold(ds, data_dir: Path, out_dir: Path, s: dict, train_idx: list[int],
                   use_depth: bool, client=None) -> dict:
    """Decide the scaffold (Claude or automatic) and attach it to the dataset."""
    if not s["scaffold"]:
        return {"enabled": False}
    if not (use_depth and ds.prior_dir is not None):
        log.info("Scaffold skipped: it needs the depth prior (to see where surfaces are visible)")
        return {"enabled": False, "why": "no depth prior"}
    try:
        import scaffold_agent
        surfaces, info = scaffold_agent.build(
            ds, s, out_dir, ds.load_prior_raw, train_idx, scaffold_up(ds),
            scene_name=str(s.get("_scene_name") or data_dir.name), client=client)
        live = scaffold_mod.Scaffold(surfaces, s)
        if len(live):
            ds.scaffold = live
            ds.scaffold_conf = float(s["scaffold_conf"])
        ds.scaffold_all = surfaces
        return {"enabled": bool(len(live)), **info}
    except Exception as e:  # an optional extra: never lose a run over it
        log.warning("Scaffold skipped: %s: %s", type(e).__name__, e)
        ds.scaffold = None
        return {"enabled": False, "error": f"{type(e).__name__}: {e}"}


def train(data_dir: Path, out_dir: Path, s: dict):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Install a CUDA build of torch "
                           "(see README) and check nvidia-smi.")
    device = torch.device("cuda")
    s["cap_max"] = resolve_cap_max(s["cap_max"], torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(s["seed"])
    np.random.seed(s["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = int(s["iterations"])

    # ---- data -------------------------------------------------------------
    views = None
    intr_file = data_dir / "intrinsics.json"
    if intr_file.exists():
        views = json.loads(intr_file.read_text()).get("views")
    views = views or s.get("_views")
    ds = ColmapDataset(data_dir, downscale=s["downscale"], cache=s["cache_images"],
                       align=bool(s["align"]), views=views, up_axis=s["up_axis"],
                       world_scale=float(s["world_scale"]))
    if ds.alignment is not None:
        st = ds.align_stats
        log.info("Aligned scene upright (%s up, scale x%g) from %d images; camera 'up' "
                 "agreement: median %.1f deg, 95%% within %.1f deg", st["up_axis"],
                 st["world_scale"], st["images_used"], st["up_deviation_median_deg"],
                 st["up_deviation_p95_deg"])
        if st["up_deviation_p95_deg"] > 8:
            log.warning("Cameras disagree about which way is up by more than 8 deg. Either the "
                        "360 export was not horizon-levelled, or the reconstruction itself is bent "
                        "(often reflections from glass/water). Check the splat for curved walls.")
        (out_dir / "alignment.json").write_text(json.dumps(
            {"world_to_aligned": ds.alignment.tolist(), **st}, indent=2))
    elif s["align"]:
        log.warning("Scene not aligned: %s", (ds.align_stats or {}).get("error"))

    train_idx, test_idx, expo_src = split_train_test(ds, int(s["test_every"]))
    use_depth = bool(s["depth_prior"])
    depth_summary = None
    if use_depth:
        try:
            depth_summary = depth_prior.prepare(ds, data_dir, s, train_idx)
        except Exception as e:  # the prior is an optional extra: never lose a run over it
            log.warning("Depth prior unavailable, training without it.\n%s: %s",
                        type(e).__name__, e)
            use_depth, ds.prior_dir = False, None
            s = dict(s, depth_prior=False)
            depth_summary = {"enabled": False, "error": f"{type(e).__name__}: {e}"}
    log.info("Dataset: %d images (%dx%d) -> %d train / %d held out, %d SfM points, "
             "scene scale %.3f", len(ds), ds.frames[0]["W"], ds.frames[0]["H"],
             len(train_idx), len(test_idx), len(ds.points), ds.scene_scale)

    # ---- scaffold: the big surfaces as geometric rules ----------------------
    scaffold_info = build_scaffold(ds, data_dir, out_dir, s, train_idx, use_depth)

    # ---- model ------------------------------------------------------------
    sh_degree = int(s["sh_degree"])
    n_sh = (sh_degree + 1) ** 2
    lrs = {"means": 1.6e-4 * ds.scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}

    def make_optimizers(splats):
        return {k: torch.optim.Adam([{"params": splats[k], "lr": lrs[k], "name": k}], eps=1e-15)
                for k in lrs}

    mcmc = s["strategy"] == "mcmc"
    refine_stop = min(25000, int(iters * 0.85)) if mcmc else min(15000, iters // 2)
    ent_end = float(s.get("opacity_entropy_end", 0.0) or 0.0)
    if ent_end <= 0:  # auto: only while MCMC still recycles what it pushes to ~0
        ent_end = refine_stop / max(iters - 1, 1) if mcmc else 1.0
    if mcmc:
        strategy = MCMCStrategy(cap_max=int(s["cap_max"]), refine_stop_iter=refine_stop)
        init_opa, init_scale = 0.5, 0.1
    elif s["strategy"] == "default":
        strategy = DefaultStrategy(refine_stop_iter=refine_stop)
        init_opa, init_scale = 0.1, 1.0
    else:
        raise ValueError(f"Unknown strategy {s['strategy']}")

    compat_keys = ("align", "up_axis", "world_scale", "exposure", "test_every", "depth_prior",
                   "bilagrid", "antialiased")
    ckpt_path = out_dir / "ckpt.pt"
    start, ckpt = 0, None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        old = ckpt.get("settings", {})
        if ckpt["step"] >= iters - 1:
            log.info("Existing checkpoint is a finished run; starting fresh")
            ckpt = None
        elif any(old.get(k) != s.get(k) for k in compat_keys) or \
                ckpt.get("n_images") != len(ds):
            log.info("Checkpoint is from different data or settings; starting fresh")
            ckpt = None

    fill_info: dict = {}
    sky_geom = None  # (centre, radius) of the sky shell, for pruning
    if ckpt is not None:
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in ckpt["splats"].items()}).to(device)
        start = ckpt["step"] + 1
        sky_geom = ckpt.get("sky_geom")
        fill_info = ckpt.get("fill_info", {})
        log.info("Resuming from step %d (%d Gaussians)", start, len(splats["means"]))
    else:
        rng = np.random.default_rng(int(s["seed"]))
        pts, cols = ds.points.astype(np.float32), ds.colors.astype(np.float32)
        seed_pts, seed_cols = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        if s["depth_seed"] and use_depth:
            seed_pts, seed_cols, fill_info["depth_seed"] = scene_fill.depth_seed_points(
                ds, train_idx, s, rng)
        sky = None
        if s["sky_shell"] and not mcmc:
            log.info("Sky shell skipped: it needs strategy mcmc (the default strategy would "
                     "prune the large distant Gaussians)")
        elif s["sky_shell"] and int(s["sky_points"]) > 0:
            n_req = min(int(s["sky_points"]), int(0.1 * int(s["cap_max"])))  # <=10% of budget
            *sky, centre, radius, fill_info["sky_shell"] = scene_fill.sky_shell_points(
                ds, train_idx, dict(s, sky_points=n_req), rng)
            sky_geom = (np.asarray(centre, np.float64), float(radius))
        n_sky = len(sky[0]) if sky else 0
        if mcmc:  # the initial set must fit the Gaussian budget
            room = max(1000, int(s["cap_max"]) - n_sky)
            if len(pts) + len(seed_pts) > room:
                frac = room / (len(pts) + len(seed_pts))
                if len(pts):
                    sel = rng.choice(len(pts), max(1, int(len(pts) * frac)), replace=False)
                    pts, cols = pts[sel], cols[sel]
                if len(seed_pts):
                    sel = rng.choice(len(seed_pts), int(len(seed_pts) * frac), replace=False)
                    seed_pts, seed_cols = seed_pts[sel], seed_cols[sel]
        body = np.concatenate([pts, seed_pts])
        body_cols = np.concatenate([cols, seed_cols])
        dist = knn_scale(body) * init_scale
        opa = np.full(len(body), init_opa, np.float32)
        all_pts, all_cols = body, body_cols
        if sky:
            sky_pts, sky_cols, sky_spacing = sky
            all_pts = np.concatenate([body, sky_pts])
            all_cols = np.concatenate([body_cols, sky_cols])
            dist = np.concatenate([dist, sky_spacing * 0.6])  # neighbours overlap: no gaps
            opa = np.concatenate([opa, np.full(n_sky, 0.8, np.float32)])
        n = len(all_pts)
        colors = torch.zeros((n, n_sh, 3))
        colors[:, 0, :] = (torch.from_numpy(all_cols.astype(np.float32)) - 0.5) / SH_C0
        splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.from_numpy(all_pts.astype(np.float32))),
            "scales": torch.nn.Parameter(
                torch.log(torch.from_numpy(dist).float())[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.rand((n, 4))),
            "opacities": torch.nn.Parameter(torch.logit(torch.from_numpy(opa))),
            "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
            "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
        }).to(device)
        log.info("Initialised %d Gaussians: %d from SfM points, %d depth seeds, %d sky shell",
                 n, len(pts), len(seed_pts), n_sky)

    optimizers = make_optimizers(splats)
    flat = (scaffold_mod.FlatTerm(ds.scaffold, device)
            if ds.scaffold is not None and float(s["scaffold_flat_weight"]) > 0 else None)
    n_surf = len(ds.scaffold) if ds.scaffold is not None else 0
    surf_vis = torch.zeros(max(n_surf, 1), dtype=torch.float64, device=device)
    surf_bad = torch.zeros(max(n_surf, 1), dtype=torch.float64, device=device)
    flat_members = 0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / iters))
    strategy.check_sanity(splats, optimizers)
    try:
        state = strategy.initialize_state(scene_scale=ds.scene_scale)
    except TypeError:
        state = strategy.initialize_state()

    # Per-image exposure: a learned 3x3 colour matrix + offset per photo, stored as the
    # difference from identity. Absorbs auto-exposure / white-balance shifts so the
    # Gaussians don't have to (which otherwise shows up as blur and floaters).
    use_expo = bool(s["exposure"])
    expo = torch.nn.Parameter(torch.zeros(len(ds), 3, 4, device=device))
    expo_opt = torch.optim.Adam([expo], lr=float(s["exposure_lr"]), eps=1e-15)
    eye3 = torch.eye(3, device=device)
    bil, bil_opt = appearance.make(len(ds), s, device)
    train_idx_t = torch.tensor(train_idx, device=device)

    # Camera refinement: one correction per 360 frame (all its views share it: rig stays rigid)
    frame_keys = []
    for f_ in ds.frames:
        pv = naming.parse(f_["name"])
        frame_keys.append(pv[0] if pv else f_["name"])
    uniq = sorted(set(frame_keys))
    fidx_of = {k: j for j, k in enumerate(uniq)}
    frame_of = torch.tensor([fidx_of[k] for k in frame_keys], device=device)
    use_pose = bool(s["pose_opt"])
    pose_adj = torch.nn.Parameter(torch.zeros(len(uniq), 9, device=device))
    pose_opt = torch.optim.Adam([pose_adj], lr=float(s["pose_lr"]),
                                weight_decay=float(s["pose_reg"]))
    rot6_id = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=device)

    def pose_matrix(j):
        """4x4 world-space correction for frame j (identity at zero)."""
        a = pose_adj[j]
        r6 = a[3:] + rot6_id
        b1 = F.normalize(r6[:3], dim=0)
        b2 = F.normalize(r6[3:] - (b1 * r6[3:]).sum() * b1, dim=0)
        Rm = torch.stack([b1, b2, torch.linalg.cross(b1, b2)], 0)
        T = torch.eye(4, device=device)
        T = T.clone()
        T[:3, :3] = Rm
        T[:3, 3] = a[:3]
        return T
    if bil is not None:
        log.info("Per-photo bilateral grids on (%dx%dx%d)", int(s["bilagrid_xy"]),
                 int(s["bilagrid_xy"]), int(s["bilagrid_luma"]))

    if ckpt is not None:
        for k, o in optimizers.items():
            o.load_state_dict(ckpt["optimizers"][k])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["strategy_state"]
        if "expo" in ckpt:
            with torch.no_grad():
                expo.copy_(ckpt["expo"])
            expo_opt.load_state_dict(ckpt["expo_opt"])
        if bil is not None and "bilagrid" in ckpt:
            bil.load_state_dict(ckpt["bilagrid"])
            bil_opt.load_state_dict(ckpt["bilagrid_opt"])
        if "pose_adj" in ckpt and ckpt["pose_adj"].shape == pose_adj.shape:
            with torch.no_grad():
                pose_adj.copy_(ckpt["pose_adj"])
            pose_opt.load_state_dict(ckpt["pose_opt"])
        del ckpt

    def apply_expo(img, i):  # img [1,H,W,3]: the photo-specific look (not exported)
        if i is None:
            return img
        if use_expo:
            d = expo[i]
            img = img @ (eye3 + d[:, :3]).T + d[:, 3]
        if bil is not None:
            img = bil(img, i)
        return img

    ssim = make_ssim(torch, F, device)
    absgrad = bool(getattr(strategy, "absgrad", False))
    packed = bool(s["packed"])
    lam = float(s["ssim_lambda"])

    def view_of(fr, i=None):
        V = torch.from_numpy(fr["viewmat"]).to(device)
        if use_pose and i is not None:
            V = V @ pose_matrix(frame_of[i])
        return V

    def render(fr, sh_deg, with_depth=False, i=None):
        return rasterization(
            render_mode="RGB+ED" if with_depth else "RGB",
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=view_of(fr, i)[None],
            Ks=torch.from_numpy(fr["K"]).to(device)[None],
            width=fr["W"], height=fr["H"], sh_degree=sh_deg,
            near_plane=float(s["near_plane"]), packed=packed, absgrad=absgrad,
            rasterize_mode="antialiased" if s["antialiased"] else "classic")

    eval_dir = out_dir / "eval"
    eval_history = []

    @torch.no_grad()
    def evaluate(step, save_images, raw=False, record=True):
        """Held-out PSNR. raw=True skips the per-photo corrections: what a viewer shows.
        record=False: a trial score (cleanup), kept out of eval_history."""
        if not test_idx:
            return None
        if save_images:
            eval_dir.mkdir(exist_ok=True)
            pick = set(test_idx[:: max(1, len(test_idx) // 6)][:6])
        psnrs = []
        for i in test_idx:
            img, _ = ds.load(i)
            gt = torch.from_numpy(img).to(device).float().div_(255.0)[None]
            out, _, _ = render(ds.frames[i], sh_degree, i=i)
            out = (out[..., :3] if raw else apply_expo(out[..., :3], expo_src.get(i))).clamp(0, 1)
            mse = F.mse_loss(out, gt).item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
            if save_images and i in pick:
                pair = torch.cat([gt[0], out[0]], dim=1).mul(255).byte().cpu().numpy()
                name = naming.flat_key(ds.frames[i]["name"])
                cv2.imwrite(str(eval_dir / f"{name}_photo_vs_splat.jpg"),
                            cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        val = float(np.mean(psnrs))
        if raw:
            log.info("EVAL step %d: held-out PSNR %.2f without per-photo corrections", step, val)
            return val
        if record:
            eval_history.append({"step": step, "psnr": round(val, 3)})
        log.info("EVAL step %d: held-out PSNR %.2f over %d unseen images%s", step, val,
                 len(psnrs), "" if record else " (trial)")
        return val

    def save_ckpt(step):
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"step": step, "n_images": len(ds),
                    "splats": {k: v.detach() for k, v in splats.items()},
                    "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                    "scheduler": scheduler.state_dict(), "strategy_state": state,
                    "expo": expo.detach(), "expo_opt": expo_opt.state_dict(),
                    **({"bilagrid": bil.state_dict(), "bilagrid_opt": bil_opt.state_dict()}
                       if bil is not None else {}),
                    "sky_geom": sky_geom, "fill_info": fill_info,
                    "pose_adj": pose_adj.detach(), "pose_opt": pose_opt.state_dict(),
                    "settings": {k: v for k, v in s.items() if not k.startswith("_")}}, tmp)
        tmp.replace(ckpt_path)

    # ---- training loop ----------------------------------------------------
    loader = Prefetcher(ds, train_idx, threads=int(s["loader_threads"]), seed=s["seed"] + start)
    t0, t_last, psnr_acc, n_acc = time.time(), time.time(), 0.0, 0
    depth_acc, depth_n = 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    eval_every = int(s["eval_every"])
    try:
        for step in range(start, iters):
            i, img, mask, prior = loader.get()
            fr = ds.frames[i]
            gt = torch.from_numpy(img).to(device, non_blocking=True).float().div_(255.0)[None]

            want_depth = use_depth and prior is not None
            renders, alphas, info = render(fr, min(step // int(s["sh_degree_interval"]), sh_degree),
                                           with_depth=want_depth,
                                           i=i if step >= int(s["pose_opt_start"]) else None)
            out = renders[..., :3]
            if s["random_background"]:
                out = out + torch.rand(1, 1, 1, 3, device=device) * (1.0 - alphas)
            out = apply_expo(out, i)

            if mask is not None:
                m = torch.from_numpy(mask).to(device).float()[None, ..., None]
                out_l, gt_l = out * m, gt * m
                l1 = (out - gt).abs().mul(m).sum() / (m.sum() * 3).clamp(min=1)
            else:
                out_l, gt_l = out, gt
                l1 = F.l1_loss(out, gt)
            loss = (1 - lam) * l1 + lam * (1 - ssim(out_l.permute(0, 3, 1, 2),
                                                     gt_l.permute(0, 3, 1, 2)))
            if mcmc:
                loss = loss + s["opacity_reg"] * torch.sigmoid(splats["opacities"]).abs().mean()
                loss = loss + s["scale_reg"] * torch.exp(splats["scales"]).abs().mean()
            frac_t = step / max(iters - 1, 1)
            ent_start = float(s["opacity_entropy_start"])
            if float(s["opacity_entropy"]) > 0 and ent_start <= frac_t < ent_end:
                w_ent = float(s["opacity_entropy"]) * min(1.0, (frac_t - ent_start) / 0.1)
                oc = torch.sigmoid(splats["opacities"]).clamp(1e-4, 1 - 1e-4)
                loss = loss - w_ent * (oc * oc.log() + (1 - oc) * (1 - oc).log()).mean()
            if use_expo:
                loss = loss + float(s["exposure_reg"]) * expo[i].pow(2).mean()
            if bil is not None:
                loss = loss + float(s["bilagrid_tv"]) * bil.tv_loss(i)
            if want_depth:
                frac = step / max(iters - 1, 1)
                lam_d = float(s["depth_weight"]) * (1 - frac * (1 - float(s["depth_end_ratio"])))
                pd = torch.from_numpy(prior[0]).to(device, non_blocking=True)
                pw = torch.from_numpy(prior[1]).to(device, non_blocking=True)
                valid = (pd > 0) & (alphas[0, ..., 0].detach() > 0.5)
                if mask is not None:
                    valid = valid & torch.from_numpy(mask).to(device)
                d_loss = depth_loss(renders[0, ..., 3], pd, pw, valid,
                                    float(s["depth_grad_weight"]))
                loss = loss + lam_d * d_loss
                depth_acc += float(d_loss.detach())
                depth_n += 1
                if prior[2] is not None and float(s["scaffold_weight"]) > 0:
                    lim = torch.from_numpy(prior[2]).to(device, non_blocking=True)
                    if mask is not None:
                        lim = lim * torch.from_numpy(mask).to(device)
                    rd = renders[0, ..., 3]
                    loss = loss + float(s["scaffold_weight"]) * scaffold_mod.free_space_loss(rd, lim)
                    if step % 10 == 0:  # how often each surface's empty-space rule fires
                        with torch.no_grad():
                            sid = torch.from_numpy(prior[3]).to(device).long()
                            on = (lim > 0) & (sid < 255)
                            ids_on = sid[on]
                            surf_vis += torch.bincount(ids_on, minlength=n_surf)[:n_surf].double()
                            bad = (rd.detach()[on] < lim[on])
                            surf_bad += torch.bincount(ids_on[bad], minlength=n_surf)[:n_surf].double()
            if flat is not None and step >= int(s["scaffold_flat_start"]):
                if flat.n_at != len(splats["means"]) or step % int(s["scaffold_flat_every"]) == 0:
                    flat_members = flat.refresh(splats["means"].detach())
                loss = loss + float(s["scaffold_flat_weight"]) * flat.loss(
                    splats["quats"], splats["scales"])

            strategy.step_pre_backward(params=splats, optimizers=optimizers, state=state,
                                       step=step, info=info)
            loss.backward()
            for o in optimizers.values():
                o.step()
                o.zero_grad(set_to_none=True)
            if use_expo:
                expo_opt.step()
                expo_opt.zero_grad(set_to_none=True)
            if bil is not None:
                bil_opt.step()
                bil_opt.zero_grad(set_to_none=True)
                bil.recenter_(train_idx_t)
            if use_pose and step >= int(s["pose_opt_start"]):
                pose_opt.step()
                pose_opt.zero_grad(set_to_none=True)
            if use_expo:  # same for the global corrections: average = no change
                with torch.no_grad():
                    expo[train_idx_t] -= expo[train_idx_t].mean(0, keepdim=True)
            scheduler.step()
            if mcmc:
                strategy.step_post_backward(params=splats, optimizers=optimizers, state=state,
                                            step=step, info=info, lr=scheduler.get_last_lr()[0])
            else:
                strategy.step_post_backward(params=splats, optimizers=optimizers, state=state,
                                            step=step, info=info, packed=packed)

            with torch.no_grad():
                mse = F.mse_loss(out.clamp(0, 1), gt).item()
                psnr_acc += -10 * math.log10(max(mse, 1e-10))
                n_acc += 1

            if (step + 1) % int(s["log_every"]) == 0 or step == iters - 1:
                now = time.time()
                rate = n_acc / max(now - t_last, 1e-6)
                eta = (iters - step - 1) / max(rate, 1e-6)
                dtxt = f"  depth {depth_acc / depth_n:.4f}" if depth_n else ""
                log.info("step %6d/%d  loss %.4f  psnr %.2f%s  gaussians %d  %.1f it/s  "
                         "vram %.1f GB  eta %.0f min",
                         step + 1, iters, loss.item(), psnr_acc / n_acc, dtxt,
                         len(splats["means"]), rate, torch.cuda.max_memory_allocated() / 1e9,
                         eta / 60)
                t_last, psnr_acc, n_acc = now, 0.0, 0
                depth_acc, depth_n = 0.0, 0
            if eval_every > 0 and (step + 1) % eval_every == 0 and step < iters - 1:
                evaluate(step + 1, save_images=False)
            if (step + 1) % int(s["save_every"]) == 0 and step < iters - 1:
                save_ckpt(step)
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("GPU ran out of memory. Lower train.custom.cap_max, raise "
                           "train.custom.downscale, or close other GPU apps.")
    finally:
        loader.close()

    save_ckpt(iters - 1)  # the checkpoint keeps every Gaussian; the export is cleaned
    if s["prune"]:
        # Cleanup must never cost quality: score before, try full then light cleanup, and
        # keep the first that stays within prune_max_psnr_drop of the unpruned score.
        with torch.no_grad():
            base = evaluate(iters, save_images=False, record=False)
            original = {k: v.detach().clone() for k, v in splats.items()}
            tol = float(s["prune_max_psnr_drop"])
            fill_info["prune"] = {"before_psnr": base, "applied": "none"}
            tried = []
            for level in ("full", "light"):
                keep, counts = scene_fill.prune_mask(
                    original["means"].cpu().numpy(), original["opacities"].cpu().numpy(),
                    original["scales"].cpu().numpy(),
                    np.array([f["center"] for f in ds.frames]), ds.scene_scale, s,
                    *(sky_geom if sky_geom else (None, None)), level=level)
                if counts["removed"] == 0:
                    continue
                if any(np.array_equal(keep, k) for k in tried):
                    log.info("Floater cleanup (%s): same as a level already tried; skipped", level)
                    continue
                tried.append(keep)
                keep_t = torch.from_numpy(keep).to(device)
                for k in list(splats.keys()):
                    splats[k] = torch.nn.Parameter(original[k][keep_t])
                score = evaluate(iters, save_images=False, record=False)
                if base is None or score is None or score >= base - tol:
                    fill_info["prune"] = {**counts, "level": level, "before_psnr": base,
                                          "after_psnr": score, "applied": level}
                    log.info("Floater cleanup kept (%s): held-out PSNR %.2f -> %.2f",
                             level, base or 0, score or 0)
                    break
                log.info("Floater cleanup (%s) undone: held-out PSNR fell %.2f -> %.2f",
                         level, base, score)
                for k in list(splats.keys()):
                    splats[k] = torch.nn.Parameter(original[k].clone())
            del original
    final_eval = evaluate(iters, save_images=True)  # scored on what is exported
    raw_eval = evaluate(iters, save_images=False, raw=True)

    # ---- camera refinement record (the Fixer step and the bundle reuse it) ----
    pose_info = {"enabled": use_pose}
    if use_pose:
        with torch.no_grad():
            mats = torch.stack([pose_matrix(j) for j in range(len(uniq))]).cpu().numpy()
        trans = np.linalg.norm(mats[:, :3, 3], axis=1) / max(ds.scene_scale, 1e-9)
        rot = np.array([rot_angle_deg(m[:3, :3]) for m in mats])
        pose_info.update({"frames": len(uniq),
                          "translation_mean_pct_scene": round(float(trans.mean() * 100), 4),
                          "translation_max_pct_scene": round(float(trans.max() * 100), 4),
                          "rotation_mean_deg": round(float(rot.mean()), 4),
                          "rotation_max_deg": round(float(rot.max()), 4)})
        (out_dir / "pose_adjust.json").write_text(json.dumps(
            {"note": "world-space correction per 360 frame: viewmat' = viewmat @ M",
             "frames": {k: mats[j].tolist() for j, k in enumerate(uniq)}}))
        log.info("Camera refinement: mean shift %.3f%% of scene, mean turn %.3f deg",
                 pose_info["translation_mean_pct_scene"], pose_info["rotation_mean_deg"])
    if ds.scaffold is not None:
        vis_np, bad_np = surf_vis.cpu().numpy(), surf_bad.cpu().numpy()
        for k, srf in enumerate(ds.scaffold.surfaces):
            srf["usage"] = {"pixels_checked": int(vis_np[k]),
                            "share_in_front": round(float(bad_np[k] / max(vis_np[k], 1)), 5)}
        scaffold_info["flat_splats"] = int(flat_members)
        scaffold_info["usage"] = [{"id": x["id"], "rule": x["rule"], **x["usage"]}
                                  for x in ds.scaffold.surfaces]
        try:
            centres = np.array([f["center"] for f in ds.frames])
            scaffold_mod.save_report(ds.scaffold_all, ds.points, centres, scaffold_up(ds),
                                     out_dir, {"source": scaffold_info.get("source"),
                                               "agent": scaffold_info.get("agent"),
                                               "flat_splats": int(flat_members)})
        except Exception as e:  # noqa: BLE001
            log.warning("Couldn't update the scaffold report: %s", e)
        log.info("Scaffold at the end: %d splats lie flat on surfaces; empty-space rule still "
                 "violated on %.2f%% of checked pixels", flat_members,
                 100 * float(bad_np[:n_surf].sum() / max(vis_np[:n_surf].sum(), 1)))
    with torch.no_grad():
        opa = torch.sigmoid(splats["opacities"])
        faint_share = round(float(((opa > 0.005) & (opa < 0.3)).float().mean()), 4)
        opacity_shares = {"below_0.005": round(float((opa <= 0.005).float().mean()), 4),
                          "0.005-0.3": faint_share,
                          "above_0.3": round(float((opa >= 0.3).float().mean()), 4)}
    log.info("Opacity of the exported Gaussians: %.1f%% near-invisible (<0.005), %.1f%% "
             "see-through (0.005-0.3), %.1f%% solid (>=0.3)",
             opacity_shares["below_0.005"] * 100, faint_share * 100,
             opacity_shares["above_0.3"] * 100)

    # ---- fly-through video ---------------------------------------------------
    if s["flythrough"]:
        try:
            with torch.no_grad():
                render_flythrough(ds, lambda V, K, W, H: rasterization(
                    means=splats["means"], quats=splats["quats"],
                    scales=torch.exp(splats["scales"]),
                    opacities=torch.sigmoid(splats["opacities"]),
                    colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                    viewmats=torch.from_numpy(V).to(device)[None],
                    Ks=torch.from_numpy(K).to(device)[None], width=W, height=H,
                    sh_degree=sh_degree, near_plane=float(s["near_plane"]),
                    packed=bool(s["packed"]),
                    rasterize_mode="antialiased" if s["antialiased"] else "classic",
                )[0][0, ..., :3],
                    {k: mats[fidx_of[k]] for k in uniq} if use_pose else {},
                    float(s["flythrough_offset"]), out_dir / "flythrough.mp4")
        except Exception as e:  # noqa: BLE001  (a video is a nice-to-have)
            log.warning("Fly-through video skipped: %s: %s", type(e).__name__, e)
    with torch.no_grad():
        kept, dropped = write_ply(
            out_dir / "splat.ply",
            splats["means"].cpu().numpy(), splats["sh0"].cpu().numpy(),
            splats["shN"].cpu().numpy(), splats["opacities"].cpu().numpy(),
            splats["scales"].cpu().numpy(), splats["quats"].cpu().numpy())
    stats = {"iterations": iters, "gaussians": kept, "dropped_non_finite": dropped,
             "minutes": round((time.time() - t0) / 60, 1),
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "images_train": len(train_idx), "images_heldout": len(test_idx),
             "heldout_psnr": round(final_eval, 3) if final_eval is not None else None,
             "heldout_psnr_raw": round(raw_eval, 3) if raw_eval is not None else None,
             "eval_history": eval_history,
             "depth_prior": depth_summary,
             "fill": fill_info,
             "faint_share": faint_share,
             "opacity_shares": opacity_shares,
             "scaffold": scaffold_info,
             "pose_refine": pose_info,
             "settings": {k: v for k, v in s.items() if not k.startswith("_")}}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Wrote %s (%d Gaussians)", out_dir / "splat.ply", kept)
    if final_eval is not None:
        log.info("Held-out PSNR %.2f. Side-by-side photo vs splat images: %s", final_eval, eval_dir)


def apply_pose_adjust(ds: "ColmapDataset", path: Path) -> int:
    """Apply a saved camera refinement (pose_adjust.json) to a dataset's cameras."""
    adj = json.loads(Path(path).read_text()).get("frames") or {}
    n = 0
    for f in ds.frames:
        pv = naming.parse(f["name"])
        key = pv[0] if pv else f["name"]
        if key in adj:
            V = (f["viewmat"].astype(np.float64) @ np.asarray(adj[key], np.float64))
            f["viewmat"] = V.astype(np.float32)
            f["center"] = (-V[:3, :3].T @ V[:3, 3]).astype(f["center"].dtype)
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Fly-through video
# --------------------------------------------------------------------------- #
def flythrough_path(ds: "ColmapDataset", adjust: dict, offset: float, sub: int = 3,
                    W: int = 960, H: int = 540, hfov: float = 90.0):
    """Camera poses along the flight path (front view of each 360 frame, in frame order),
    shifted sideways and up by `offset` x scene scale, so the video shows the splat from
    places no photo was taken. Returns (list of 4x4 viewmats, K)."""
    from scipy.spatial.transform import Rotation, Slerp
    keyed = []
    for f in ds.frames:
        pv = naming.parse(f["name"])
        if pv and pv[1] == 0:
            V = f["viewmat"].astype(np.float64)
            if pv[0] in adjust:
                V = V @ np.asarray(adjust[pv[0]], np.float64)
            keyed.append((naming.frame_number(pv[0]), V))
    if len(keyed) < 2:  # no rig naming: use every image in name order
        keyed = [(j, f["viewmat"].astype(np.float64))
                 for j, f in enumerate(sorted(ds.frames, key=lambda f: f["name"]))]
    keyed.sort(key=lambda kv: kv[0])
    Rs = np.stack([V[:3, :3] for _, V in keyed])
    Cs = np.stack([-V[:3, :3].T @ V[:3, 3] for _, V in keyed])
    # gentle smoothing so drone jitter doesn't dominate the video
    k = 5
    pad = np.pad(Cs, ((k // 2, k // 2), (0, 0)), mode="edge")
    Cs = np.stack([pad[i:i + k].mean(0) for i in range(len(Cs))])
    n = len(Rs)
    ts = np.linspace(0, n - 1, (n - 1) * sub + 1)
    slerp = Slerp(np.arange(n), Rotation.from_matrix(Rs))
    R_all = slerp(ts).as_matrix()
    C_all = np.stack([np.interp(ts, np.arange(n), Cs[:, d]) for d in range(3)], 1)
    sh = offset * ds.scene_scale
    out = []
    for R, C in zip(R_all, C_all):
        C2 = C + R[0] * sh - R[1] * sh * 0.5   # camera right, and up (camera y points down)
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = -R @ C2
        out.append(V.astype(np.float32))
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    return out, K


def render_flythrough(ds, render_rgb, adjust: dict, offset: float, path: Path,
                      W: int = 960, H: int = 540, fps: int = 30):
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff is None:
        log.warning("Fly-through video skipped: ffmpeg not found")
        return
    views, K = flythrough_path(ds, adjust, offset, W=W, H=H)
    t0 = time.time()
    proc = subprocess.Popen(
        [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)
    try:
        for V in views:
            img = render_rgb(V, K, W, H).clamp(0, 1).mul(255).byte().cpu().numpy()
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc}")
    log.info("Fly-through video: %s (%d frames, %.1f s, made in %.0f s)", path, len(views),
             len(views) / fps, time.time() - t0)


# --------------------------------------------------------------------------- #
# Inspect (no GPU needed)
# --------------------------------------------------------------------------- #
def rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def inspect(data_dir: Path, s: dict):
    """Checks how self-consistent the reconstruction is. All views sliced from one 360
    frame were really taken from the same point with known relative angles, so any
    disagreement between them is SfM error, which shows up as warping and blur."""
    sparse = data_dir / "sparse" / "0"
    images = read_images_txt(sparse / "images.txt")
    views = None
    intr_file = data_dir / "intrinsics.json"
    if intr_file.exists():
        views = json.loads(intr_file.read_text()).get("views")
    views = views or s.get("_views")
    if not views:
        raise ValueError("No view list found (intrinsics.json or config slice.views)")

    frames: dict[str, list] = {}
    for im in images:
        parsed = naming.parse(im["name"])
        if not parsed or parsed[1] >= len(views):
            continue
        world_from_rig = im["R"].T @ rig_from_cam(*views[parsed[1]][:2]).T
        frames.setdefault(parsed[0], []).append((parsed[1], -im["R"].T @ im["t"], world_from_rig))

    centers_all = np.array([c for f in frames.values() for _, c, _ in f])
    scene = float(np.median(np.linalg.norm(centers_all - centers_all.mean(0), axis=1))) or 1.0
    pos_err, rot_err, per_view = [], [], {}
    for f in frames.values():
        if len(f) < 2:
            continue
        c_mean = np.mean([c for _, c, _ in f], axis=0)
        Rs = [R for _, _, R in f]
        U, _, Vt = np.linalg.svd(sum(Rs))  # rotation average
        R_mean = U @ Vt
        for vi, c, R in f:
            pe = np.linalg.norm(c - c_mean) / scene * 100
            re_ = rot_angle_deg(R_mean.T @ R)
            pos_err.append(pe)
            rot_err.append(re_)
            per_view.setdefault(vi, []).append(re_)

    M, st = compute_alignment(images, views, s["up_axis"])
    print(f"Registered images: {len(images)} in {len(frames)} frames "
          f"(avg {len(images) / max(len(frames), 1):.1f} of {len(views)} views per frame)")
    print(f"Up-direction agreement: median {st['up_deviation_median_deg']:.2f} deg, "
          f"95% within {st['up_deviation_p95_deg']:.2f} deg")
    print(f"Views of the same frame - position spread: median {np.median(pos_err):.2f}%, "
          f"95% within {np.percentile(pos_err, 95):.2f}% of flight radius")
    print(f"Views of the same frame - angle error:     median {np.median(rot_err):.2f} deg, "
          f"95% within {np.percentile(rot_err, 95):.2f} deg")
    print("Angle error by view (median deg): " + "  ".join(
        f"v{vi:02d}={np.median(e):.2f}({len(e)})" for vi, e in sorted(per_view.items())))
    bad = np.percentile(rot_err, 95) > 2 or np.percentile(pos_err, 95) > 5
    print("Verdict: " + ("INCONSISTENT - SfM error is likely causing warping/blur. Use the rig "
                         "constraint and/or mask reflective surfaces." if bad else
                         "consistent - geometry is sound; blur is a training/capture issue."))


# --------------------------------------------------------------------------- #
# Self check
# --------------------------------------------------------------------------- #
def depth_preview(data_dir: Path, s: dict):
    s = dict(s, depth_prior=True)
    views = None
    intr_file = data_dir / "intrinsics.json"
    if intr_file.exists():
        views = json.loads(intr_file.read_text()).get("views")
    views = views or s.get("_views")
    ds = ColmapDataset(data_dir, downscale=s["downscale"], align=bool(s["align"]), views=views,
                       up_axis=s["up_axis"], world_scale=float(s["world_scale"]))
    train_idx, _, _ = split_train_test(ds, int(s["test_every"]))
    summary = depth_prior.prepare(ds, data_dir, s, train_idx)
    out = data_dir / "depth_prior" / "preview"
    depth_prior.preview(ds, Path(summary["cache"]), out)
    log.info("Preview images (photo | aligned prior | weight, white = supervised): %s", out)


def scaffold_only(data_dir: Path, out_dir: Path, s: dict) -> dict:
    """Just the scaffold step (depth priors + candidates + Claude), no training."""
    s = dict(s, depth_prior=True, scaffold=True)
    views = None
    intr_file = data_dir / "intrinsics.json"
    if intr_file.exists():
        views = json.loads(intr_file.read_text()).get("views")
    views = views or s.get("_views")
    ds = ColmapDataset(data_dir, downscale=s["downscale"], align=bool(s["align"]), views=views,
                       up_axis=s["up_axis"], world_scale=float(s["world_scale"]))
    train_idx, _, _ = split_train_test(ds, int(s["test_every"]))
    depth_prior.prepare(ds, data_dir, s, train_idx)
    info = build_scaffold(ds, data_dir, out_dir, s, train_idx, True)
    (out_dir / "scaffold_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def self_check():
    import torch
    log.info("torch %s, CUDA build %s", torch.__version__, torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False: torch is a CPU build or the "
                           "driver is too old")
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("Importing gsplat (the first run compiles CUDA kernels; this can take several minutes)")
    import gsplat
    from gsplat import rasterization
    log.info("gsplat %s", getattr(gsplat, "__version__", "?"))
    dev = torch.device("cuda")
    n = 2000
    means = (torch.rand(n, 3, device=dev) - 0.5) * 2 + torch.tensor([0, 0, 4.0], device=dev)
    means.requires_grad_(True)
    out, _, _ = rasterization(
        means=means, quats=torch.rand(n, 4, device=dev), scales=torch.full((n, 3), 0.05, device=dev),
        opacities=torch.full((n,), 0.8, device=dev), colors=torch.rand(n, 3, device=dev),
        viewmats=torch.eye(4, device=dev)[None],
        Ks=torch.tensor([[[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]]], device=dev),
        width=128, height=128, rasterize_mode="antialiased")
    out.mean().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    log.info("OK: rendered and back-propagated through gsplat on the GPU")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    child = os.environ.get("SPLAT_CHILD") == "1"  # launched by splat_pipeline.py, which timestamps
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt="%H:%M:%S",
                        format="%(message)s" if child else "%(asctime)s %(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description="Train a Gaussian splat from a COLMAP dataset")
    ap.add_argument("--data", type=Path, help="dataset folder containing images/ and sparse/0/")
    ap.add_argument("--out", type=Path, help="output folder for splat.ply and checkpoints")
    ap.add_argument("--config", help="config.yaml (reads train.iterations and train.custom)")
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--strategy", choices=["mcmc", "default"])
    ap.add_argument("--cap-max", type=int, dest="cap_max")
    ap.add_argument("--downscale", type=float)
    ap.add_argument("--sh-degree", type=int, dest="sh_degree")
    ap.add_argument("--check", action="store_true", help="verify torch/CUDA/gsplat and exit")
    ap.add_argument("--inspect", action="store_true",
                    help="report how self-consistent the COLMAP model is (no GPU needed)")
    ap.add_argument("--depth-preview", action="store_true", dest="depth_preview",
                    help="compute depth priors only and save photo|prior|weight images")
    ap.add_argument("--scaffold-only", action="store_true", dest="scaffold_only",
                    help="build the scene scaffold only (Claude or automatic) into --out")
    args = ap.parse_args()

    try:
        if args.check:
            self_check()
            return
        if args.inspect:
            if not args.data:
                ap.error("--inspect needs --data")
            inspect(args.data.resolve(), load_settings(args.config or None, {}))
            return
        if args.depth_preview:
            if not args.data:
                ap.error("--depth-preview needs --data")
            depth_preview(args.data.resolve(), load_settings(args.config or None, {}))
            return
        if not args.data or not args.out:
            ap.error("--data and --out are required (or use --check / --inspect)")
        if args.scaffold_only:
            args.out.mkdir(parents=True, exist_ok=True)
            scaffold_only(args.data.resolve(), args.out.resolve(),
                          load_settings(args.config or None, {}))
            return
        overrides = {k: getattr(args, k) for k in ("iterations", "strategy", "cap_max",
                                                    "downscale", "sh_degree")}
        settings = load_settings(args.config or None, overrides)
        log.info("Settings: %s", json.dumps(settings))
        train(args.data.resolve(), args.out.resolve(), settings)
    except Exception as e:
        log.error("%s: %s", type(e).__name__, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
