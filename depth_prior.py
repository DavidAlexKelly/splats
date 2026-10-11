"""
depth_prior.py - monocular depth priors for Gaussian splat training.

Pipeline (all cached under <data>/depth_prior/<backend tag>/):

 1. Predict a relative depth map for every training image with a monocular model
    (Depth Anything 3 mono if installed, else Depth Anything V2 Small via transformers).
    Both default models are Apache-2.0.
 2. Align each map to the SfM points that image actually observed: fit scale + shift
    robustly, trying both "depth" and "inverse depth" spaces and keeping the better fit
    (so we don't have to know which one a given model outputs). Images whose fit is poor
    get no depth supervision at all.
 3. Build a per-pixel "geometry is weak here" weight from the density of SfM observations:
    areas full of tracked points are already pinned down by multi-view geometry, so the
    prior is only applied lightly there; sparse/empty areas get full weight. This follows
    the 2026 finding that naive depth supervision everywhere can hurt well-reconstructed
    regions ("In Depth We Trust", arXiv 2604.05715).

The trainer then adds a disparity L1 + depth-gradient loss weighted by that map.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("train")

DEFAULT_MODELS = {
    "da3": "da3mono-large",                              # Apache-2.0
    "da2": "depth-anything/Depth-Anything-V2-Small-hf",  # Apache-2.0
}


# --------------------------------------------------------------------------- #
# Alignment (pure numpy, unit-tested)
# --------------------------------------------------------------------------- #
def robust_affine(x: np.ndarray, y: np.ndarray, iters: int = 4, keep: float = 0.8):
    """Fit y ~ a*x + b, re-fitting on the best `keep` fraction of points each round."""
    idx = np.arange(len(x))
    a = b = 0.0
    for _ in range(iters):
        A = np.stack([x[idx], np.ones(len(idx))], 1)
        sol, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        res = np.abs(a * x + b - y)
        n = max(8, int(len(x) * keep))
        idx = np.argsort(res)[:n]
    return a, b


def sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample a HxW map at float pixel coords (pixel centres at +0.5)."""
    H, W = img.shape
    x = np.clip(u - 0.5, 0, W - 1.001)
    y = np.clip(v - 0.5, 0, H - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def align_to_sfm(rel: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray,
                 min_points: int = 30, max_rel_error: float = 0.35):
    """Align a relative map to sparse metric-ish depths z observed at pixels (u, v).
    Returns (depth map in scene units or None, info dict)."""
    ok = (z > 0) & np.isfinite(z)
    u, v, z = u[ok], v[ok], z[ok]
    info = {"points": int(len(z))}
    if len(z) < min_points:
        info["rejected"] = "too few SfM points"
        return None, info
    r = sample(rel.astype(np.float64), u, v)
    best = None
    for space in ("inverse", "depth"):
        target = 1.0 / z if space == "inverse" else z
        a, b = robust_affine(r, target)
        if a <= 0:
            continue
        pred = a * r + b
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = 1.0 / pred if space == "inverse" else pred
        good = (zp > 0) & np.isfinite(zp)
        if good.sum() < min_points:
            continue
        err = float(np.median(np.abs(zp[good] - z[good]) / z[good]))
        if best is None or err < best[0]:
            best = (err, space, a, b)
    if best is None:
        info["rejected"] = "no positive fit"
        return None, info
    err, space, a, b = best
    # A prior must beat "everything is at the median depth" clearly, or it carries no shape
    baseline = float(np.median(np.abs(np.median(z) - z) / z))
    info.update(space=space, rel_error=round(err, 4), flat_baseline=round(baseline, 4))
    if err > max_rel_error:
        info["rejected"] = f"fit error {err:.2f} > {max_rel_error}"
        return None, info
    if baseline > 0.05 and err > 0.6 * baseline:
        info["rejected"] = f"prior explains little depth variation ({err:.2f} vs flat {baseline:.2f})"
        return None, info
    m = a * rel.astype(np.float64) + b
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 1.0 / m if space == "inverse" else m
    zmin, zmax = np.percentile(z, 1) * 0.3, np.percentile(z, 99) * 4.0
    depth = np.where(np.isfinite(depth) & (depth > zmin), np.minimum(depth, zmax), 0.0)
    return depth.astype(np.float32), info


def coverage_weight(u: np.ndarray, v: np.ndarray, H: int, W: int, mode: str = "selective",
                    min_weight: float = 0.15, sigma_frac: float = 0.04) -> np.ndarray:
    """1 where SfM observations are sparse (geometry weak), down to min_weight where dense."""
    if mode == "uniform":
        return np.ones((H, W), np.float32)
    s = 64  # work on a coarse grid, then upsample
    gh, gw = max(4, H * s // max(H, W)), max(4, W * s // max(H, W))
    grid = np.zeros((gh, gw), np.float32)
    gx = np.clip((u / W * gw).astype(int), 0, gw - 1)
    gy = np.clip((v / H * gh).astype(int), 0, gh - 1)
    np.add.at(grid, (gy, gx), 1.0)
    sig = max(0.8, sigma_frac * max(gh, gw))
    dens = cv2.GaussianBlur(grid, (0, 0), sig)
    ref = np.percentile(dens[dens > 0], 75) if (dens > 0).any() else 1.0
    w = 1.0 - np.clip(dens / max(ref, 1e-6), 0, 1)
    w = np.clip(w, min_weight, 1.0).astype(np.float32)
    return cv2.resize(w, (W, H), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Monocular depth models
# --------------------------------------------------------------------------- #
class _DA3:
    def __init__(self, model: str, process_res: int):
        import torch
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            from depth_anything_3 import DepthAnything3
        name = model or DEFAULT_MODELS["da3"]
        self.m = (DepthAnything3.from_pretrained(name) if "/" in name
                  else DepthAnything3(model_name=name)).to("cuda").eval()
        self.res = process_res
        self.torch = torch

    def __call__(self, paths: list[Path]) -> list[np.ndarray]:
        with self.torch.no_grad():
            pred = self.m.inference(image=[str(p) for p in paths], process_res=self.res)
        d = pred.depth
        d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
        return [x.astype(np.float32) for x in d]


class _DA2:
    def __init__(self, model: str, process_res: int):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = model or DEFAULT_MODELS["da2"]
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.m = AutoModelForDepthEstimation.from_pretrained(name).to("cuda").eval()
        self.torch = torch

    def __call__(self, paths: list[Path]) -> list[np.ndarray]:
        imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        inputs = self.proc(images=imgs, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = self.m(**inputs).predicted_depth  # relative inverse depth
        return [x.float().cpu().numpy() for x in out]


def load_model(backend: str, model: str, process_res: int):
    errors = []
    for b in (["da3", "da2"] if backend == "auto" else [backend]):
        try:
            m = (_DA3 if b == "da3" else _DA2)(model if backend != "auto" else "", process_res)
            return b, m
        except Exception as e:  # missing package / weights / network
            errors.append(f"{b}: {type(e).__name__}: {e}")
    raise RuntimeError("No depth model could be loaded:\n  " + "\n  ".join(errors) +
                       "\nInstall transformers (pip install transformers) or Depth Anything 3, "
                       "or set train.custom.depth_prior: false")


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def frame_observations(fr: dict, points: np.ndarray, scale_x: float, scale_y: float):
    """(u, v, z) at training resolution for the SfM points this image observed."""
    uv, pidx = fr.get("obs_uv"), fr.get("obs_pidx")
    if uv is None or len(pidx) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    X = points[pidx]
    V = fr["viewmat"]
    z = X @ V[2, :3] + V[2, 3]
    return uv[:, 0] * scale_x, uv[:, 1] * scale_y, z


def prepare(ds, data_dir: Path, s: dict, train_idx: list[int]) -> dict:
    """Compute + cache aligned priors and weights for the training images."""
    tag = f"{s['depth_backend']}_{(s['depth_model'] or 'default').replace('/', '_')}_" \
          f"{s['depth_mode']}_d{s['downscale']}_ws{s['world_scale']}_a{int(bool(s['align']))}"
    cache = data_dir / "depth_prior" / tag
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in train_idx if not (cache / f"{Path(ds.frames[i]['name']).stem}.npz").exists()
            and not (cache / f"{Path(ds.frames[i]['name']).stem}.skip").exists()]
    used = None
    if todo:
        used, model = load_model(s["depth_backend"], s["depth_model"], int(s["depth_process_res"]))
        log.info("Depth prior: predicting %d images with %s", len(todo), used)
        bs = int(s["depth_batch"])
        for k in range(0, len(todo), bs):
            chunk = todo[k:k + bs]
            rels = model([ds.frames[i]["path"] for i in chunk])
            for i, rel in zip(chunk, rels):
                fr = ds.frames[i]
                H, W = fr["H"], fr["W"]
                rel = cv2.resize(rel, (W, H), interpolation=cv2.INTER_LINEAR)
                u, v, z = frame_observations(fr, ds.points, W / fr["src_w"], H / fr["src_h"])
                depth, info = align_to_sfm(rel, u, v, z, int(s["depth_min_points"]),
                                           float(s["depth_max_rel_error"]))
                stem = Path(fr["name"]).stem
                if depth is None:
                    (cache / f"{stem}.skip").write_text(json.dumps(info))
                    continue
                w = coverage_weight(u, v, H, W, s["depth_mode"], float(s["depth_min_weight"]))
                np.savez_compressed(cache / f"{stem}.npz", depth=depth.astype(np.float16),
                                    weight=(w * 255).astype(np.uint8),
                                    info=json.dumps(info))
            if (k // bs) % 20 == 0:
                log.info("Depth prior: %d/%d", min(k + bs, len(todo)), len(todo))
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    have = sum((cache / f"{Path(ds.frames[i]['name']).stem}.npz").exists() for i in train_idx)
    skipped = sum((cache / f"{Path(ds.frames[i]['name']).stem}.skip").exists() for i in train_idx)
    summary = {"cache": str(cache), "with_prior": have, "rejected": skipped,
               "train_images": len(train_idx), "backend_used": used}
    log.info("Depth prior: %d/%d training images supervised (%d rejected for poor SfM fit)",
             have, len(train_idx), skipped)
    if have < 0.5 * len(train_idx):
        log.warning("Depth prior covers under half the images; check the fit errors in %s", cache)
    ds.prior_dir = cache
    return summary


def load(cache: Path | None, name: str):
    if cache is None:
        return None
    f = cache / f"{Path(name).stem}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["depth"].astype(np.float32), z["weight"].astype(np.float32) / 255.0


def preview(ds, cache: Path, out_dir: Path, n: int = 6):
    """Side-by-side photo | aligned prior | weight images for a quick visual check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("*.npz"))
    for f in files[:: max(1, len(files) // n)][:n]:
        fr = next(x for x in ds.frames if Path(x["name"]).stem == f.stem)
        img, _ = ds.load(ds.frames.index(fr))
        d, w = load(cache, fr["name"])
        valid = d > 0
        inv = np.zeros_like(d)
        inv[valid] = 1.0 / d[valid]
        lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0, 1)
        dvis = cv2.applyColorMap((np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
                                 .astype(np.uint8), cv2.COLORMAP_TURBO)
        wvis = cv2.cvtColor((w * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        row = np.concatenate([cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dvis, wvis], 1)
        cv2.imwrite(str(out_dir / f"{f.stem}_photo_prior_weight.jpg"), row,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
