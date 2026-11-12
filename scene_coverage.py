"""
How well the real photos cover each part of a splat: where its gaps are.

Every Gaussian gets a coverage score from 0 (no real photo saw it properly) to 1 (seen by
enough photos, close enough and from different enough directions to pin it down):

  - project it into a spread of training cameras;
  - it counts as seen by a camera when it lands inside the frame and isn't hidden behind
    the surface that camera's render shows there (rendered depth);
  - each sighting counts less the further away it was (fewer pixels on it);
  - score = min(1, sightings / coverage_good_views) x min(1, angle between the sightings /
    coverage_good_spread). One view, or many views from one direction, can't pin a point.

For a NEW viewpoint that score is the wrong question: from a flight path a few metres across,
nothing far away is seen from 20 degrees apart, yet a view a little off the path sees a far
wall from much the same direction as the photos did, so the photos already show it. So each
Gaussian also records which directions real photos saw it from (63 direction bins), and
Coverage.for_view(camera) says, per Gaussian, whether a real photo saw it from about the same
direction as that camera. That is what the Fixer step treats as a gap.

Used by
  train_gsplat.py   coverage_topdown.jpg + coverage.json with every result: a map of the
                    poorly covered areas (red) and the worst few numbered (fly there next time)
  fixer_refine.py   aims the AI fill-in at the gaps and only lets it change poorly covered
                    pixels (a coverage image rendered from each new viewpoint).

Empty space is a gap too: a pixel where the splat renders nothing gets coverage 0.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("train")

DEFAULTS = {
    "coverage": True,
    "coverage_cameras": 256,       # training cameras used to measure coverage
    "coverage_good_views": 6,      # sightings (close-up equivalent) for full coverage
    "coverage_good_spread": 20.0,  # degrees between viewing directions for full coverage
    "coverage_low": 0.3,           # below this = poorly covered (red on the map)
    "coverage_depth_scale": 0.25,  # depth maps for the visibility test, x photo resolution
}


def setting(s: dict, k: str):
    return s.get(k, DEFAULTS[k])


class SplatView:
    """Renders the live splat (a dict of parameter tensors) with any colours."""

    def __init__(self, splats, s: dict, sh_degree: int, antialiased: bool):
        self.splats, self.s, self.sh_degree = splats, s, sh_degree
        self.mode = "antialiased" if antialiased else "classic"

    def raster(self, fr: dict, colors=None, depth: bool = False, scale: float = 1.0):
        """(image [H,W,C], alpha [H,W]) for a frame dict (viewmat, K, W, H)."""
        import torch
        from gsplat import rasterization
        sp = self.splats
        dev = sp["means"].device
        K = torch.as_tensor(np.asarray(fr["K"]), dtype=torch.float32, device=dev).clone()
        W, H = int(fr["W"]), int(fr["H"])
        if scale != 1.0:
            K[:2] *= scale
            W, H = max(8, int(W * scale)), max(8, int(H * scale))
        sh = None if colors is not None else self.sh_degree
        cols = colors if colors is not None else torch.cat([sp["sh0"], sp["shN"]], 1)
        with torch.no_grad():  # measuring only, never trained through
            r, a, _ = rasterization(
                means=sp["means"], quats=sp["quats"], scales=torch.exp(sp["scales"]),
                opacities=torch.sigmoid(sp["opacities"]), colors=cols,
                viewmats=torch.as_tensor(np.asarray(fr["viewmat"]), dtype=torch.float32,
                                         device=dev)[None],
                Ks=K[None], width=W, height=H, sh_degree=sh,
                render_mode="RGB+ED" if depth else "RGB",
                near_plane=float(self.s.get("near_plane", 0.01)),
                packed=bool(self.s.get("packed", True)), rasterize_mode=self.mode)
        return r[0], a[0, ..., 0]

    def depth(self, fr: dict, scale: float):
        """Expected depth where the render is solid (alpha > 0.5), else 0."""
        import torch
        r, a = self.raster(fr, depth=True, scale=scale)
        d = r[..., 3]
        return torch.where(a > 0.5, d, torch.zeros_like(d))

    def coverage_image(self, fr: dict, cov, scale: float = 1.0):
        """Per-pixel coverage (0..1) seen from a camera: the per-Gaussian values `cov` (e.g.
        Coverage.for_view(fr)) of whatever the pixel shows, weighted by how opaque it is;
        empty pixels are 0."""
        import torch
        cols = cov.float()[:, None].expand(-1, 3).contiguous()
        img, _ = self.raster(fr, colors=cols, scale=scale)
        return img[..., 0].clamp(0, 1)


def _observe(P, fr: dict, D, near: float):
    """(seen by this camera, camera-z) for world points P [M,3]."""
    import torch
    dev = P.device
    V = torch.as_tensor(np.asarray(fr["viewmat"]), dtype=torch.float32, device=dev)
    K = torch.as_tensor(np.asarray(fr["K"]), dtype=torch.float32, device=dev)
    Xc = P @ V[:3, :3].T + V[:3, 3]
    z = Xc[:, 2]
    zc = z.clamp(min=1e-6)
    u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
    W, H = float(fr["W"]), float(fr["H"])
    inside = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    h, w = D.shape
    ui = (u * (w / W)).long().clamp(0, w - 1)
    vi = (v * (h / H)).long().clamp(0, h - 1)
    dd = D[vi, ui]
    # seen = nothing solid rendered there, or the point is at/in front of what is rendered
    seen = inside & ((dd <= 0) | (z <= dd * 1.05 + 1e-6))
    return seen, z


def _sphere_bins(n: int, device):
    """n roughly evenly spread unit directions (Fibonacci sphere)."""
    import torch
    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi),
                        torch.cos(phi)], 1)


N_BINS = 63  # ~26 degrees apart: a sighting counts for directions within ~15 degrees of it


def _bin_of(dirs, bins, chunk: int = 1_000_000):
    import torch
    out = torch.empty(len(dirs), dtype=torch.long, device=dirs.device)
    for k in range(0, len(dirs), chunk):
        out[k:k + chunk] = (dirs[k:k + chunk] @ bins.T).argmax(1)
    return out


class Coverage:
    """score: per-Gaussian coverage (0..1, for the map). for_view(cam): per-Gaussian 1/0,
    seen by a real photo from about the direction of that camera."""

    def __init__(self, score, dirmask, P, bins):
        self.score, self.dirmask, self.P, self.bins = score, dirmask, P, bins

    def for_view(self, cam: dict):
        import torch
        with torch.no_grad():
            c = torch.as_tensor(np.asarray(cam["center"], np.float32), device=self.P.device)
            b = _bin_of(torch.nn.functional.normalize(c[None] - self.P, dim=1), self.bins)
            return ((self.dirmask >> b) & 1).float()


def gaussian_coverage(view: SplatView, frames: list[dict], cam_idx: list[int], s: dict):
    """(Coverage, small info dict) for the splat's Gaussians."""
    import torch
    sp = view.splats
    with torch.no_grad():
        P = sp["means"].detach().float()
        N = len(P)
        dev = P.device
        scale = float(setting(s, "coverage_depth_scale"))
        near = float(s.get("near_plane", 0.01))
        cams = list(cam_idx)
        if len(cams) > int(setting(s, "coverage_cameras")):
            pick = np.linspace(0, len(cams) - 1, int(setting(s, "coverage_cameras")))
            cams = [cams[int(k)] for k in pick]
        # typical sighting distance, from a quick look with a few cameras
        sub = torch.randperm(N, device=dev)[: min(N, 200_000)]
        zs = []
        for i in cams[:: max(1, len(cams) // 24)]:
            fr = frames[i]
            seen, z = _observe(P[sub], fr, view.depth(fr, scale), near)
            if seen.any():
                zs.append(z[seen])
        z_ref = float(torch.cat(zs).median()) if zs else 1.0
        n = torch.zeros(N, device=dev)
        dsum = torch.zeros(N, 3, device=dev)
        bins = _sphere_bins(N_BINS, dev)
        dirmask = torch.zeros(N, dtype=torch.long, device=dev)
        for i in cams:
            fr = frames[i]
            seen, z = _observe(P, fr, view.depth(fr, scale), near)
            if not seen.any():
                continue
            w = torch.where(seen, (z_ref / z.clamp(min=1e-6)).pow(2).clamp(max=1.0),
                            torch.zeros_like(z))
            c = torch.as_tensor(np.asarray(fr["center"], np.float32), device=dev)
            dirs = torch.nn.functional.normalize(c[None] - P, dim=1)
            n += w
            dsum += w[:, None] * dirs
            bit = torch.bitwise_left_shift(torch.ones_like(dirmask), _bin_of(dirs, bins))
            dirmask |= torch.where(seen, bit, torch.zeros_like(bit))
        mean_len = (dsum.norm(dim=1) / n.clamp(min=1e-6)).clamp(0, 1)
        spread = torch.rad2deg(2 * torch.acos(mean_len))  # ~ full angle between sightings
        cov = ((n / float(setting(s, "coverage_good_views"))).clamp(max=1)
               * (spread / float(setting(s, "coverage_good_spread"))).clamp(max=1))
        cov = torch.where(n > 0, cov, torch.zeros_like(cov))
    return Coverage(cov, dirmask, P, bins), {"cameras": len(cams),
                                             "typical_distance": round(z_ref, 4)}


# --------------------------------------------------------------------------- #
# Report: map + numbers
# --------------------------------------------------------------------------- #
def _colour(c: np.ndarray) -> np.ndarray:
    """0 red -> 0.5 yellow -> 1 green (BGR, uint8)."""
    c = np.clip(c, 0, 1)[:, None]
    red, yel, grn = np.array([40, 40, 220.]), np.array([40, 210, 240.]), np.array([70, 170, 60.])
    lo = red + (yel - red) * np.clip(c * 2, 0, 1)
    hi = yel + (grn - yel) * np.clip(c * 2 - 1, 0, 1)
    return np.where(c < 0.5, lo, hi).astype(np.uint8)


def gap_areas(means: np.ndarray, cov: np.ndarray, up: np.ndarray, low: float,
              cams: np.ndarray, k: int = 5, grid: int = 25) -> list[dict]:
    """The worst poorly covered areas seen from above: cells with many solid Gaussians and low
    average coverage, best first, at least two cells apart."""
    import scaffold as sc
    a, _ = sc._basis(up)
    b = np.cross(up, a)
    P2 = np.stack([means @ a, means @ b], 1)
    lo, hi = np.percentile(P2, 1, axis=0), np.percentile(P2, 99, axis=0)
    cell = float(max(hi - lo)) / grid
    if cell <= 0:
        return []
    ij = np.floor((P2 - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < grid).all(1)
    cnt = np.zeros((grid, grid))
    tot = np.zeros((grid, grid))
    np.add.at(cnt, (ij[ok, 0], ij[ok, 1]), 1)
    np.add.at(tot, (ij[ok, 0], ij[ok, 1]), cov[ok])
    mean = tot / np.maximum(cnt, 1)
    score = np.where((cnt >= max(20, cnt[cnt > 0].mean() * 0.25 if (cnt > 0).any() else 20))
                     & (mean < low), cnt * (low - mean), 0)
    out = []
    C2 = np.stack([cams @ a, cams @ b], 1)
    while len(out) < k and score.max() > 0:
        i, j = np.unravel_index(np.argmax(score), score.shape)
        centre2 = lo + (np.array([i, j]) + 0.5) * cell
        sel = ok & (ij[:, 0] == i) & (ij[:, 1] == j)
        centre3 = means[sel].mean(0)
        out.append({"rank": len(out) + 1, "centre": np.round(centre3, 3).tolist(),
                    "mean_coverage": round(float(mean[i, j]), 3), "gaussians": int(cnt[i, j]),
                    "distance_to_flight_path": round(float(
                        np.min(np.linalg.norm(C2 - centre2, axis=1))), 3),
                    "_xy": centre2})
        score[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = 0
    return out


def draw_topdown(means: np.ndarray, cov: np.ndarray, cams: np.ndarray, up: np.ndarray,
                 gaps: list[dict], size: int = 1000) -> np.ndarray:
    """Top-down coverage map (BGR): each pixel coloured by the average coverage of the
    Gaussians above it; flight path dark blue; the worst gaps numbered."""
    import cv2
    import scaffold as sc
    a, _ = sc._basis(up)
    b = np.cross(up, a)
    P2 = np.stack([means @ a, means @ b], 1)
    C2 = np.stack([cams @ a, cams @ b], 1)
    lo, hi = np.percentile(P2, 1, axis=0), np.percentile(P2, 99, axis=0)
    lo, hi = np.minimum(lo, C2.min(0)), np.maximum(hi, C2.max(0))
    sc_ = (size - 60) / max(float(max(hi - lo)), 1e-9)
    px = ((P2 - lo) * sc_ + 30).astype(int)
    ok = (px >= 0).all(1) & (px < size).all(1)
    cnt = np.zeros((size, size))
    tot = np.zeros((size, size))
    np.add.at(cnt, (px[ok, 1], px[ok, 0]), 1)
    np.add.at(tot, (px[ok, 1], px[ok, 0]), cov[ok])
    # spread each point over a few pixels so the map reads as areas, not dust
    k = np.ones((5, 5), np.float32)
    cnt_b = cv2.filter2D(cnt.astype(np.float32), -1, k)
    tot_b = cv2.filter2D(tot.astype(np.float32), -1, k)
    img = np.full((size, size, 3), 255, np.uint8)
    has = cnt_b > 0.5
    img[has] = _colour((tot_b[has] / cnt_b[has]))
    for x, y in ((C2 - lo) * sc_ + 30).astype(int):
        cv2.circle(img, (int(x), int(y)), 2, (120, 40, 0), -1)
    for g in gaps:
        x, y = ((g["_xy"] - lo) * sc_ + 30).astype(int)
        cv2.circle(img, (int(x), int(y)), 18, (0, 0, 0), 2)
        sc._label(img, str(g["rank"]), int(x) - 6, int(y) + 6, 0.7)
    cv2.putText(img, "coverage from above: red = poorly covered by real photos, green = well "
                "covered; blue = flight path; numbers = worst gaps", (10, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def report(view: SplatView, frames: list[dict], cam_idx: list[int], s: dict, out_dir: Path,
           up: np.ndarray, inner_radius: float | None = None) -> dict:
    """Coverage of a finished splat: coverage_topdown.jpg, coverage.json, and the info dict."""
    import cv2
    import torch
    cov_obj, info = gaussian_coverage(view, frames, cam_idx, s)
    cov = cov_obj.score
    with torch.no_grad():
        means = view.splats["means"].detach().cpu().numpy().astype(np.float64)
        opa = torch.sigmoid(view.splats["opacities"]).detach().cpu().numpy()
    covn = cov.cpu().numpy()
    cams = np.array([frames[i]["center"] for i in cam_idx], np.float64)
    solid = opa >= 0.3
    if inner_radius:
        solid &= np.linalg.norm(means - cams.mean(0), axis=1) < inner_radius
    low = float(setting(s, "coverage_low"))
    gaps = gap_areas(means[solid], covn[solid], up, low, cams)
    info.update({
        "solid_gaussians": int(solid.sum()),
        "poorly_covered_share": round(float((covn[solid] < low).mean()), 4) if solid.any() else None,
        "mean_coverage": round(float(covn[solid].mean()), 4) if solid.any() else None,
        "worst_gaps": [{k: v for k, v in g.items() if not k.startswith("_")} for g in gaps]})
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cv2.imwrite(str(out_dir / "coverage_topdown.jpg"),
                    draw_topdown(means[solid], covn[solid], cams, up, gaps),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    except Exception as e:  # noqa: BLE001  (a picture is a nice-to-have)
        log.warning("Couldn't draw the coverage map: %s", e)
    (out_dir / "coverage.json").write_text(json.dumps(info, indent=2))
    log.info("Coverage: %.1f%% of the solid Gaussians are poorly covered by real photos "
             "(score < %.2f); worst gaps marked on coverage_topdown.jpg",
             100 * (info["poorly_covered_share"] or 0), low)
    return info


def gap_weight(cov_img: np.ndarray, lo: float, hi: float, floor: float,
               blur_px: int = 15) -> np.ndarray:
    """Per-pixel weight for an AI-filled view: 1 where coverage <= lo (a gap), `floor` where
    coverage >= hi (real photos already pin it down), smooth in between."""
    import cv2
    w = np.clip((hi - cov_img) / max(hi - lo, 1e-6), 0, 1)
    if blur_px > 1:
        k = blur_px | 1
        w = cv2.GaussianBlur(w.astype(np.float32), (k, k), 0)
    return np.maximum(w, floor).astype(np.float32)
