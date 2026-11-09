"""
Scene scaffold: the big flat surfaces of a scene (floor, walls, fences), found in the SfM
point cloud, used as geometric rules during training.

Stage 1 of the "scaffold" idea (no AI agent yet):

  detect_planes()   RANSAC on the SfM points: horizontal and vertical planes that are dense,
                    genuinely flat (points hug the plane, unlike a slab cut through a bush) and
                    cover a real area. Each gets a label (floor / wall / ceiling) and extents.
  Scaffold.depth()  per image: the depth of the nearest plane along every pixel ray.
  Scaffold.visible  where that plane is actually what the photo shows: the image's own depth
                    prior (already aligned to the SfM scale) agrees with the plane depth. A bush
                    in front of a wall disagrees, so it's never treated as empty space.

The trainer uses this twice:
  * flat-surface depth: where a plane is visible, the depth prior is replaced by the exact
    plane depth (with high confidence);
  * empty space (the floater killer): on those pixels nothing may sit in front of the plane,
    so rendered depth well in front of it is penalised for the whole of training. Floaters
    are exactly Gaussians in empty air that the photos alone can't rule out.

numpy only (testable without a GPU).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("train")


def _fit_plane(pts: np.ndarray):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n / np.linalg.norm(n), c


def _basis(n: np.ndarray):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def detect_planes(points: np.ndarray, up: np.ndarray, cam_centres: np.ndarray, scale: float,
                  s: dict, seed: int = 0) -> list[dict]:
    """Structural planes in the point cloud. `up` is the world up direction, `scale` the
    scene scale (camera path radius; minimum plane size is relative to it)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, np.float64)
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    n_all = len(pts)
    # SfM points get less precise with distance, so the plane thickness is relative to how
    # far the surfaces typically are from the cameras (not to the size of the flight path).
    sub = pts[rng.choice(n_all, min(n_all, 20000), replace=False)]
    cams = np.asarray(cam_centres, np.float64)
    cams = cams[rng.choice(len(cams), min(len(cams), 300), replace=False)]
    near = np.min(np.linalg.norm(sub[:, None, :] - cams[None, :, :], axis=2), axis=1)
    typical = float(np.median(near))
    tol = float(s["scaffold_tol"]) * typical
    min_in = max(200, int(float(s["scaffold_min_frac"]) * n_all))
    max_tilt = math.radians(float(s["scaffold_max_tilt_deg"]))
    remaining = np.arange(n_all)
    planes: list[dict] = []
    rejected = 0
    for _ in range(int(s["scaffold_max_planes"]) * 3):
        if len(planes) >= int(s["scaffold_max_planes"]) or len(remaining) < min_in:
            break
        P = pts[remaining]
        best, best_n = None, 0
        for _h in range(int(s["scaffold_ransac_iters"])):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            cos_up = abs(n @ up)
            # structural surfaces only: horizontal (floor/ceiling) or vertical (walls)
            if not (cos_up > math.cos(max_tilt) or cos_up < math.sin(max_tilt)):
                continue
            inl = np.abs((P - a) @ n) < tol
            k = int(inl.sum())
            if k > best_n:
                best, best_n = inl, k
        if best is None or best_n < min_in:
            break
        n, c = _fit_plane(P[best])
        dist = np.abs((P - c) @ n)
        inl = dist < tol
        idx = remaining[inl]
        # A real surface is a sharp peak: most points near it are ON it. A slab cut through
        # a bush or a tree has points spread evenly through a wider band (~25% within tol).
        wide = dist < 4 * tol
        concentration = float(inl.sum() / max(int(wide.sum()), 1))
        remaining = remaining[~(dist < 2 * tol)]  # don't find the same surface twice
        q = pts[idx]
        u, v = _basis(n)
        cu, cv = (q - c) @ u, (q - c) @ v
        lo_u, hi_u = np.percentile(cu, [2, 98])
        lo_v, hi_v = np.percentile(cv, [2, 98])
        # ...and covers its extent rather than being a thin line or a few clumps
        g = 24
        iu = np.clip(((cu - lo_u) / max(hi_u - lo_u, 1e-9) * g).astype(int), 0, g - 1)
        iv = np.clip(((cv - lo_v) / max(hi_v - lo_v, 1e-9) * g).astype(int), 0, g - 1)
        occupancy = len(np.unique(iu * g + iv)) / (g * g)
        area = float((hi_u - lo_u) * (hi_v - lo_v))
        cos_up = abs(n @ up)
        if cos_up > math.cos(max_tilt):
            # floor if the cameras are above it, ceiling if below
            side = np.median((cam_centres - c) @ (n if n @ up > 0 else -n))
            label = "floor" if side > 0 else "ceiling"
        else:
            label = "wall"
        ok = (concentration > float(s["scaffold_min_concentration"])
              and occupancy > float(s["scaffold_min_occupancy"])
              and area > (0.05 * scale) ** 2)
        if not ok:
            rejected += 1
            log.info("Scaffold: skipped a %s candidate (%d points): concentration %.2f (min %.2f),"
                     " coverage %.2f (min %.2f), area %.2f (min %.2f)", label, len(idx),
                     concentration, float(s["scaffold_min_concentration"]), occupancy,
                     float(s["scaffold_min_occupancy"]), area, (0.05 * scale) ** 2)
            continue
        planes.append({"label": label, "normal": n.tolist(), "centre": c.tolist(),
                       "u": u.tolist(), "v": v.tolist(),
                       "extent": [float(lo_u), float(hi_u), float(lo_v), float(hi_v)],
                       "points": int(len(idx)), "concentration": round(concentration, 3),
                       "occupancy": round(occupancy, 3), "area": round(area, 3)})
    log.info("Scaffold: %d plane(s) (%s) from %d SfM points; %d candidate(s) rejected as not "
             "flat/solid", len(planes), ", ".join(p["label"] for p in planes) or "none",
             n_all, rejected)
    return planes


class Scaffold:
    """Plane depths per image, computed at 1/`step` resolution and cached."""

    def __init__(self, planes: list[dict], s: dict, margin_pad: float = 0.02):
        self.planes = planes
        self.step = max(1, int(s["scaffold_res_step"]))
        self.agree = float(s["scaffold_agree"])
        self.pad = margin_pad
        self._cache: dict[str, np.ndarray] = {}
        self.N = np.array([p["normal"] for p in planes], np.float64).reshape(-1, 3)
        self.C = np.array([p["centre"] for p in planes], np.float64).reshape(-1, 3)
        self.U = np.array([p["u"] for p in planes], np.float64).reshape(-1, 3)
        self.V = np.array([p["v"] for p in planes], np.float64).reshape(-1, 3)
        self.E = np.array([p["extent"] for p in planes], np.float64).reshape(-1, 4)

    def depth(self, fr: dict) -> np.ndarray:
        """Camera-z depth of the nearest plane along each (subsampled) pixel ray; 0 = none."""
        key = fr["name"]
        if key in self._cache:
            return self._cache[key]
        K, Vm = fr["K"].astype(np.float64), fr["viewmat"].astype(np.float64)
        H, W, st = fr["H"], fr["W"], self.step
        vv, uu = np.mgrid[st // 2:H:st, st // 2:W:st]
        dirs_c = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1],
                           np.ones_like(uu, np.float64)], -1)        # z = 1 in camera
        R, t = Vm[:3, :3], Vm[:3, 3]
        o = -R.T @ t                                                  # camera centre (world)
        dirs_w = dirs_c @ R                                           # R^T d
        best = np.full(uu.shape, np.inf)
        for k in range(len(self.N)):
            n = self.N[k]
            denom = dirs_w @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                z = ((self.C[k] - o) @ n) / denom                     # camera z (dirs have z=1)
            hit = o + dirs_w * z[..., None]
            pu, pv = (hit - self.C[k]) @ self.U[k], (hit - self.C[k]) @ self.V[k]
            lo_u, hi_u, lo_v, hi_v = self.E[k]
            pad_u, pad_v = self.pad * (hi_u - lo_u), self.pad * (hi_v - lo_v)
            inside = ((z > 0) & np.isfinite(z) & (pu > lo_u - pad_u) & (pu < hi_u + pad_u)
                      & (pv > lo_v - pad_v) & (pv < hi_v + pad_v))
            best = np.where(inside & (z < best), z, best)
        out = np.where(np.isfinite(best), best, 0).astype(np.float16)
        self._cache[key] = out
        return out

    def merge_prior(self, fr: dict, depth: np.ndarray, weight: np.ndarray, conf: float):
        """(depth, weight, scaffold_depth): the prior with plane depth substituted where the
        plane is what the image shows, and the plane depth there (0 elsewhere) for the
        empty-space term."""
        H, W = depth.shape
        low = self.depth(fr).astype(np.float32)
        import cv2
        plane = cv2.resize(low, (W, H), interpolation=cv2.INTER_NEAREST)
        vis = (plane > 0) & (depth > 0) & (np.abs(depth - plane) < self.agree * plane)
        if not vis.any():
            return depth, weight, None
        d2 = np.where(vis, plane, depth).astype(np.float32)
        w2 = np.where(vis, np.maximum(weight, conf), weight).astype(np.float32)
        return d2, w2, np.where(vis, plane, 0).astype(np.float32)


def free_space_loss(rend_depth, plane_depth, margin: float):
    """Penalise anything rendered clearly in front of a visible plane (empty space)."""
    import torch
    valid = plane_depth > 0
    if int(valid.sum()) < 64:
        return rend_depth.sum() * 0.0
    pd = plane_depth[valid]
    gap = torch.relu(pd * (1 - margin) - rend_depth[valid]) / pd
    return gap.mean()


def save_report(planes: list[dict], points: np.ndarray, cam_centres: np.ndarray,
                up: np.ndarray, out_dir: Path):
    """scaffold.json and a top-down picture (points grey, planes coloured, cameras red)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaffold.json").write_text(json.dumps({"planes": planes}, indent=2))
    try:
        import cv2
        up = up / np.linalg.norm(up)
        a, _ = _basis(up)
        b = np.cross(up, a)
        P2 = np.stack([points @ a, points @ b], 1)
        C2 = np.stack([cam_centres @ a, cam_centres @ b], 1)
        lo, hi = np.percentile(P2, 1, axis=0), np.percentile(P2, 99, axis=0)
        lo, hi = np.minimum(lo, C2.min(0)), np.maximum(hi, C2.max(0))
        S = 900
        sc = (S - 40) / max(hi - lo)
        to_px = lambda q: ((q - lo) * sc + 20).astype(int)  # noqa: E731
        img = np.full((S, S, 3), 255, np.uint8)
        for x, y in to_px(P2[:: max(1, len(P2) // 60000)]):
            if 0 <= x < S and 0 <= y < S:
                img[y, x] = (170, 170, 170)
        colours = {"floor": (60, 160, 60), "wall": (200, 90, 30), "ceiling": (150, 60, 160)}
        for p in planes:
            c, u, v = (np.array(p[k]) for k in ("centre", "u", "v"))
            lu, hu, lv, hv = p["extent"]
            corners = [c + u * x + v * y for x, y in ((lu, lv), (hu, lv), (hu, hv), (lu, hv))]
            q = to_px(np.array([[k @ a, k @ b] for k in corners]))
            cv2.polylines(img, [q.reshape(-1, 1, 2)], True, colours[p["label"]], 2)
        for x, y in to_px(C2):
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 220), -1)
        y = S - 12
        x = 12
        for text, col in (("floor", colours["floor"]), ("wall", colours["wall"]),
                          ("ceiling", colours["ceiling"]), ("cameras", (0, 0, 220))):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            x += 12 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.imwrite(str(out_dir / "scaffold_topdown.jpg"), img)
    except Exception as e:  # noqa: BLE001  (a picture is a nice-to-have)
        log.warning("Couldn't draw the scaffold picture: %s", e)
