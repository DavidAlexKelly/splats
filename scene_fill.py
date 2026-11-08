"""
Making the most of imperfect footage, using only real data (no generative fill):

  depth_seed_points  Extra starting Gaussians where SfM has few points: the per-image depth
                     prior (already aligned to the SfM scale) is back-projected in the regions
                     its coverage weight marks as weakly observed. Training then has something
                     to refine there instead of an empty hole.
  sky_shell_points   A sphere of Gaussians far outside the scene, coloured from the photos.
                     Open sky (and anything seen only in the far distance) is drawn on it
                     instead of rendering black or growing floaters. Saved in the .ply like
                     any other Gaussian, so every viewer shows it.
  prune_mask         Post-training floater cleanup: near-invisible Gaussians, Gaussians
                     hugging the camera path, isolated specks and oversized blobs.

numpy/scipy only, so it can be tested without a GPU.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger("train")


def _camera_centres(ds, idx) -> np.ndarray:
    return np.array([ds.frames[i]["center"] for i in idx], dtype=np.float64).reshape(-1, 3)


def _backproject(fr: dict, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pixel centres (u, v) at depth z (camera z) -> world points."""
    K, V = fr["K"].astype(np.float64), fr["viewmat"].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    cam = np.stack([x, y, z], 1)
    R, t = V[:3, :3], V[:3, 3]
    return (cam - t) @ R  # R^T (cam - t)


def _voxel_dedupe(pts: np.ndarray, cols: np.ndarray, voxel: float):
    if len(pts) == 0 or voxel <= 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return pts[first], cols[first]


def depth_seed_points(ds, train_idx: list[int], s: dict, rng: np.random.Generator):
    """Returns (points Nx3, colours Nx3 in 0..1, info)."""
    budget = int(s["depth_seed_max_points"])
    if budget <= 0 or ds.prior_dir is None:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"seeded": 0, "reason": "no depth prior"}
    min_w = float(s["depth_seed_min_weight"])
    stride = max(1, int(s["depth_seed_stride"]))
    with_prior = [i for i in train_idx if ds.load_prior(i) is not None]
    if not with_prior:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"seeded": 0, "reason": "no aligned priors"}
    per_image = max(50, int(2.5 * budget / len(with_prior)))  # oversample, dedupe trims it
    pts_all, cols_all = [], []
    for i in with_prior:
        depth, weight = ds.load_prior(i)
        fr = ds.frames[i]
        img, mask = ds.load(i)
        H, W = depth.shape
        vv, uu = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
        vv, uu = vv.ravel(), uu.ravel()
        ok = (depth[vv, uu] > 0) & (weight[vv, uu] >= min_w)
        if mask is not None:
            ok &= mask[vv, uu]
        vv, uu = vv[ok], uu[ok]
        if len(vv) == 0:
            continue
        if len(vv) > per_image:
            pick = rng.choice(len(vv), per_image, replace=False)
            vv, uu = vv[pick], uu[pick]
        z = depth[vv, uu].astype(np.float64)
        pts_all.append(_backproject(fr, uu + 0.5, vv + 0.5, z))
        cols_all.append(img[vv, uu].astype(np.float64) / 255.0)
    if not pts_all:
        return np.zeros((0, 3)), np.zeros((0, 3)), {"seeded": 0, "reason": "no weak regions"}
    pts, cols = np.concatenate(pts_all), np.concatenate(cols_all)
    # Keep seeds inside the region the SfM points occupy (priors can run off to infinity)
    lo, hi = np.percentile(ds.points, 0.5, axis=0), np.percentile(ds.points, 99.5, axis=0)
    pad = 0.1 * (hi - lo)
    inside = np.all((pts > lo - pad) & (pts < hi + pad), axis=1)
    pts, cols = pts[inside], cols[inside]
    pts, cols = _voxel_dedupe(pts, cols, float(s["depth_seed_voxel"]) * ds.scene_scale)
    if len(pts) > budget:
        pick = rng.choice(len(pts), budget, replace=False)
        pts, cols = pts[pick], cols[pick]
    info = {"seeded": int(len(pts)), "images": len(with_prior)}
    log.info("Depth seeding: %d extra starting Gaussians in weakly observed areas (%d images)",
             len(pts), len(with_prior))
    return pts.astype(np.float32), cols.astype(np.float32), info


def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = math.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)


def sky_shell_points(ds, train_idx: list[int], s: dict, rng: np.random.Generator):
    """Returns (points, colours 0..1, per-point radius (for initial scale), centre, radius, info)."""
    n = int(s["sky_points"])
    cams = _camera_centres(ds, train_idx)
    centre = cams.mean(0)
    d = np.linalg.norm(ds.points - centre, axis=1)
    scene_extent = float(np.percentile(d, 98)) if len(d) else ds.scene_scale
    radius = float(s["sky_radius_mult"]) * max(scene_extent, ds.scene_scale)
    dirs = _fibonacci_sphere(n)
    pts = centre + radius * dirs
    # Colour each shell point from photos that look towards it (the far background seen
    # through that direction); points no photo sees stay mid grey.
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    order = list(train_idx)
    rng.shuffle(order)
    for i in order[: min(len(order), 400)]:
        fr = ds.frames[i]
        V = fr["viewmat"].astype(np.float64)
        cam = pts @ V[:3, :3].T + V[:3, 3]
        front = cam[:, 2] > 0
        if not front.any():
            continue
        K = fr["K"]
        u = K[0, 0] * cam[front, 0] / cam[front, 2] + K[0, 2]
        v = K[1, 1] * cam[front, 1] / cam[front, 2] + K[1, 2]
        inb = (u >= 0) & (u < fr["W"] - 1) & (v >= 0) & (v < fr["H"] - 1)
        if not inb.any():
            continue
        img, mask = ds.load(i)
        idx = np.where(front)[0][inb]
        ui, vi = u[inb].astype(int), v[inb].astype(int)
        good = mask[vi, ui] if mask is not None else np.ones(len(ui), bool)
        acc[idx[good]] += img[vi[good], ui[good]].astype(np.float64) / 255.0
        cnt[idx[good]] += 1
    cols = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], 0.5)
    spacing = radius * math.sqrt(4 * math.pi / n)  # mean distance between shell points
    info = {"points": n, "radius": round(radius, 4), "coloured": int((cnt > 0).sum())}
    log.info("Sky shell: %d Gaussians on a sphere of radius %.2f (%d coloured from photos)",
             n, radius, info["coloured"])
    return (pts.astype(np.float32), cols.astype(np.float32),
            np.full(n, spacing, np.float32), centre, radius, info)


def prune_mask(means: np.ndarray, opacity_logit: np.ndarray, scale_log: np.ndarray,
               cam_centres: np.ndarray, scene_scale: float, s: dict,
               sky_centre=None, sky_radius: float | None = None, level: str = "full"):
    """Boolean keep-mask and per-rule counts.
    level "full": all rules; "light": only isolated specks and oversized blobs.
    Sky-shell Gaussians are never touched except when essentially invisible (< 0.005).
    NB with the MCMC strategy many Gaussians are faint but matter together, so the caller
    checks the held-out score and backs off if pruning hurts it."""
    from scipy.spatial import cKDTree
    n = len(means)
    keep = np.ones(n, bool)
    counts = {}
    opa = 1 / (1 + np.exp(-opacity_logit.reshape(-1)))
    sky = np.zeros(n, bool)
    if sky_centre is not None and sky_radius:
        sky = np.linalg.norm(means - sky_centre, axis=1) > 0.5 * sky_radius
    inner = ~sky
    thr = float(s["prune_min_opacity"]) if level == "full" else 0.0
    rule = (inner & (opa < thr)) | (opa < 0.005)
    counts["low_opacity"] = int(rule.sum())
    keep &= ~rule

    if level == "full" and float(s["prune_near_camera"]) > 0 and len(cam_centres):
        dist, _ = cKDTree(cam_centres).query(means, k=1)
        rule = inner & (dist < float(s["prune_near_camera"]) * scene_scale)
        counts["near_camera"] = int((rule & keep).sum())
        keep &= ~rule

    if float(s["prune_max_scale"]) > 0:
        big = np.exp(scale_log).max(axis=1) > float(s["prune_max_scale"]) * scene_scale
        rule = inner & big
        counts["oversized"] = int((rule & keep).sum())
        keep &= ~rule

    k_std = float(s["prune_outlier_std"])
    cand = np.where(keep & inner)[0]
    if k_std > 0 and len(cand) > 16:
        tree = cKDTree(means[cand])
        d, _ = tree.query(means[cand], k=9, workers=-1)
        md = d[:, 1:].mean(axis=1)
        # Robust z-score of the mean neighbour distance: isolated specks stand out
        med = np.median(md)
        mad = np.median(np.abs(md - med)) * 1.4826 + 1e-12
        rule_c = (md - med) / mad > k_std * 4  # MAD units are tighter than std on heavy tails
        counts["isolated"] = int(rule_c.sum())
        keep[cand[rule_c]] = False

    counts["kept"] = int(keep.sum())
    counts["removed"] = int(n - keep.sum())
    log.info("Floater cleanup (%s): removing %d of %d Gaussians (%s)", level, n - keep.sum(), n,
             ", ".join(f"{k} {v}" for k, v in counts.items() if k not in ("kept", "removed")))
    return keep, counts
