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
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_prior  # noqa: E402

log = logging.getLogger("train")

DEFAULTS = {
    "iterations": 30000,
    "strategy": "mcmc",          # "mcmc" (fixed Gaussian budget, predictable VRAM) or "default"
    "cap_max": 1_000_000,        # MCMC: max number of Gaussians. ~1M fits an 8 GB GPU comfortably
    "sh_degree": 3,              # view-dependent colour detail (0-3)
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
}

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def load_settings(config_path: str | None, overrides: dict) -> dict:
    s = dict(DEFAULTS)
    if config_path:
        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
        train = cfg.get("train", {}) or {}
        if "iterations" in train:
            s["iterations"] = train["iterations"]
        s.update(train.get("custom", {}) or {})
        views = (cfg.get("slice", {}) or {}).get("views")
        if views:
            s["_views"] = views
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
def rig_from_cam(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation taking camera-frame rays to the level 360 rig frame. Must match
    build_view_maps() in splat_pipeline.py: pitch about x first, then yaw about y.
    Frames use x right, y down, z forward."""
    p, t = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    return Ry @ Rx


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
        m = re.search(r"^(.*)_v(\d+)\.", im["name"])
        if not m or int(m.group(2)) >= len(views):
            continue
        frame, vi = m.group(1), int(m.group(2))
        R_rc = rig_from_cam(*views[vi][:2])
        world_from_cam = im["R"].T
        ups.append(world_from_cam @ R_rc.T @ up_rig)
        if first_frame is None:
            first_frame = frame
        if frame == first_frame:
            fwds.append(world_from_cam @ R_rc.T @ fwd_rig)
        centers.append(-im["R"].T @ im["t"])
    if len(ups) < 10:
        return None, {"error": "too few images with _vNN names to align"}
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
        centers = np.array([f["center"] for f in self.frames])
        radius = float(np.max(np.linalg.norm(centers - centers.mean(0), axis=1)))
        self.scene_scale = radius * 1.1 if radius > 0 else 1.0
        self.cache = {} if cache else None
        self.prior_dir = None  # set by depth_prior.prepare()

    def load_prior(self, i: int):
        return depth_prior.load(self.prior_dir, self.frames[i]["name"])

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
                item = (i, *self.ds.load(i), self.ds.load_prior(i))
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
    """'f00012_v03.jpg' -> (12, 3); None if the name doesn't follow the pattern."""
    m = re.search(r"(\d+)_v(\d+)\.", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


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


def train(data_dir: Path, out_dir: Path, s: dict):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch. Install a CUDA build of torch "
                           "(see README) and check nvidia-smi.")
    device = torch.device("cuda")
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
    depth_summary = depth_prior.prepare(ds, data_dir, s, train_idx) if use_depth else None
    log.info("Dataset: %d images (%dx%d) -> %d train / %d held out, %d SfM points, "
             "scene scale %.3f", len(ds), ds.frames[0]["W"], ds.frames[0]["H"],
             len(train_idx), len(test_idx), len(ds.points), ds.scene_scale)

    # ---- model ------------------------------------------------------------
    sh_degree = int(s["sh_degree"])
    n_sh = (sh_degree + 1) ** 2
    lrs = {"means": 1.6e-4 * ds.scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}

    def make_optimizers(splats):
        return {k: torch.optim.Adam([{"params": splats[k], "lr": lrs[k], "name": k}], eps=1e-15)
                for k in lrs}

    mcmc = s["strategy"] == "mcmc"
    if mcmc:
        strategy = MCMCStrategy(cap_max=int(s["cap_max"]),
                                refine_stop_iter=min(25000, int(iters * 0.85)))
        init_opa, init_scale = 0.5, 0.1
    elif s["strategy"] == "default":
        strategy = DefaultStrategy(refine_stop_iter=min(15000, iters // 2))
        init_opa, init_scale = 0.1, 1.0
    else:
        raise ValueError(f"Unknown strategy {s['strategy']}")

    compat_keys = ("align", "up_axis", "world_scale", "exposure", "test_every", "depth_prior")
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

    if ckpt is not None:
        splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in ckpt["splats"].items()}).to(device)
        start = ckpt["step"] + 1
        log.info("Resuming from step %d (%d Gaussians)", start, len(splats["means"]))
    else:
        pts, cols = ds.points, ds.colors
        if mcmc and len(pts) > s["cap_max"]:
            sel = np.random.choice(len(pts), int(s["cap_max"]), replace=False)
            pts, cols = pts[sel], cols[sel]
        n = len(pts)
        dist = knn_scale(pts) * init_scale
        colors = torch.zeros((n, n_sh, 3))
        colors[:, 0, :] = (torch.from_numpy(cols) - 0.5) / SH_C0
        splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.from_numpy(pts)),
            "scales": torch.nn.Parameter(torch.log(torch.from_numpy(dist).float())[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(torch.rand((n, 4))),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opa))),
            "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
            "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
        }).to(device)
        log.info("Initialised %d Gaussians from SfM points", n)

    optimizers = make_optimizers(splats)
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

    if ckpt is not None:
        for k, o in optimizers.items():
            o.load_state_dict(ckpt["optimizers"][k])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["strategy_state"]
        if "expo" in ckpt:
            with torch.no_grad():
                expo.copy_(ckpt["expo"])
            expo_opt.load_state_dict(ckpt["expo_opt"])
        del ckpt

    def apply_expo(img, i):  # img [1,H,W,3]
        if not use_expo or i is None:
            return img
        d = expo[i]
        return img @ (eye3 + d[:, :3]).T + d[:, 3]

    ssim = make_ssim(torch, F, device)
    absgrad = bool(getattr(strategy, "absgrad", False))
    packed = bool(s["packed"])
    lam = float(s["ssim_lambda"])

    def render(fr, sh_deg, with_depth=False):
        return rasterization(
            render_mode="RGB+ED" if with_depth else "RGB",
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]), opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=torch.from_numpy(fr["viewmat"]).to(device)[None],
            Ks=torch.from_numpy(fr["K"]).to(device)[None],
            width=fr["W"], height=fr["H"], sh_degree=sh_deg,
            near_plane=float(s["near_plane"]), packed=packed, absgrad=absgrad)

    eval_dir = out_dir / "eval"
    eval_history = []

    @torch.no_grad()
    def evaluate(step, save_images):
        if not test_idx:
            return None
        if save_images:
            eval_dir.mkdir(exist_ok=True)
            pick = set(test_idx[:: max(1, len(test_idx) // 6)][:6])
        psnrs = []
        for i in test_idx:
            img, _ = ds.load(i)
            gt = torch.from_numpy(img).to(device).float().div_(255.0)[None]
            out, _, _ = render(ds.frames[i], sh_degree)
            out = apply_expo(out[..., :3], expo_src.get(i)).clamp(0, 1)
            mse = F.mse_loss(out, gt).item()
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
            if save_images and i in pick:
                pair = torch.cat([gt[0], out[0]], dim=1).mul(255).byte().cpu().numpy()
                name = Path(ds.frames[i]["name"]).stem
                cv2.imwrite(str(eval_dir / f"{name}_photo_vs_splat.jpg"),
                            cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        val = float(np.mean(psnrs))
        eval_history.append({"step": step, "psnr": round(val, 3)})
        log.info("EVAL step %d: held-out PSNR %.2f over %d unseen images", step, val, len(psnrs))
        return val

    def save_ckpt(step):
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"step": step, "n_images": len(ds),
                    "splats": {k: v.detach() for k, v in splats.items()},
                    "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                    "scheduler": scheduler.state_dict(), "strategy_state": state,
                    "expo": expo.detach(), "expo_opt": expo_opt.state_dict(),
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
                                           with_depth=want_depth)
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
            if use_expo:
                loss = loss + float(s["exposure_reg"]) * expo[i].pow(2).mean()
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

            strategy.step_pre_backward(params=splats, optimizers=optimizers, state=state,
                                       step=step, info=info)
            loss.backward()
            for o in optimizers.values():
                o.step()
                o.zero_grad(set_to_none=True)
            if use_expo:
                expo_opt.step()
                expo_opt.zero_grad(set_to_none=True)
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

    save_ckpt(iters - 1)
    final_eval = evaluate(iters, save_images=True)
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
             "eval_history": eval_history,
             "depth_prior": depth_summary,
             "settings": {k: v for k, v in s.items() if not k.startswith("_")}}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Wrote %s (%d Gaussians)", out_dir / "splat.ply", kept)
    if final_eval is not None:
        log.info("Held-out PSNR %.2f. Side-by-side photo vs splat images: %s", final_eval, eval_dir)


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
        m = re.search(r"^(.*)_v(\d+)\.", im["name"])
        if not m or int(m.group(2)) >= len(views):
            continue
        world_from_rig = im["R"].T @ rig_from_cam(*views[int(m.group(2))][:2]).T
        frames.setdefault(m.group(1), []).append((int(m.group(2)), -im["R"].T @ im["t"], world_from_rig))

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
        width=128, height=128)
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
