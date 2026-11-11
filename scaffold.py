"""
Scene scaffold: the big surfaces of a scene (floor, walls, fences, glass doors), each with a
rule for how training treats it.

  find_candidates()   loose RANSAC on the SfM points: every flat patch worth a look, level,
                      upright or sloped, split into connected pieces. Over-finds on purpose:
                      deciding what is real is the next step's job. A surface that RANSAC
                      found as several stacked, overlapping layers (a floor a little thicker
                      or more tilted than the plane tolerance) is merged back into one.
  photo_check()       projects every candidate into a sample of photos and compares it with
                      each photo's depth estimate: how much of it the photos agree is there,
                      from how many viewpoints, how much they see THROUGH it (glass), how much
                      is hidden behind something. Photo agreement also fills in the outline
                      where a plain surface has few SfM points.
  auto_rules()        the no-AI decision (used when Claude isn't available): solid where the
                      points or the photos clearly say flat, ignore the rest.
  scaffold_agent.py   Claude reviews the candidates and decides (the default when a key is set).

Rules:
  solid   paving, walls, decking: exact depth where visible, the space in front must stay
          empty (kills floaters) and splats on it lie flat.
  glass   windows, glass doors: as solid, but "visible" comes from the outline rather than the
          depth estimate (which sees through glass), so reflections become paint on the
          surface instead of blobs floating in front of or behind it. For glass you
          mostly see reflections in.
  window  glass you mostly see THROUGH (a lit room behind a door at dusk, a shop window):
          the space in front of it must stay empty, but what is behind keeps its own depth
          (no depth override, no flattening), so the room keeps its parallax.
  soft    grass, gravel, low planting: only the space well in front must stay empty; never
          flattened.
  ignore  not used.

Each surface is a plane with a real outline: a grid of cells on the plane (not a rectangle),
so a floor doesn't spill over steps, a lawn edge or a pond.

numpy (+ OpenCV for drawing); torch only in the training-time helpers.
"""
from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger("train")

RULES = ("solid", "glass", "window", "soft", "ignore")
RULE_COLOURS = {"solid": (60, 170, 60), "glass": (230, 170, 40), "window": (200, 90, 190),
                "soft": (40, 200, 220), "ignore": (150, 150, 150)}  # BGR
SHEET_RULES = ("glass", "window")  # placed from the outline, win ties with their wall
VERSION = 2


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _fit_plane(pts: np.ndarray):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n / np.linalg.norm(n), c


def _basis(n: np.ndarray, up: np.ndarray | None = None):
    """In-plane axes. For upright planes u is horizontal and v points up."""
    if up is not None and abs(n @ up) < 0.95:
        u = np.cross(up, n)
        u /= np.linalg.norm(u)
        return u, np.cross(n, u)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def orientation(n: np.ndarray, up: np.ndarray, cams: np.ndarray, c: np.ndarray,
                tilt_deg: float = 10.0) -> str:
    cos_up = abs(float(n @ up))
    if cos_up > math.cos(math.radians(tilt_deg)):
        side = np.median((cams - c) @ (n if n @ up > 0 else -n))
        return "floor" if side > 0 else "ceiling"
    if cos_up < math.sin(math.radians(tilt_deg)):
        return "wall"
    return "slope"


def mask_to_str(m: np.ndarray) -> str:
    return base64.b64encode(np.packbits(m.astype(np.uint8).ravel()).tobytes()).decode()


def mask_from_str(s: str, shape) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))
    return bits[: shape[0] * shape[1]].reshape(shape).astype(bool)


def typical_distance(points: np.ndarray, cams: np.ndarray, rng) -> float:
    """Median distance from a surface point to the nearest camera: SfM precision and the
    size of a useful outline cell both scale with it."""
    sub = points[rng.choice(len(points), min(len(points), 20000), replace=False)]
    c = cams[rng.choice(len(cams), min(len(cams), 300), replace=False)]
    near = np.min(np.linalg.norm(sub[:, None, :] - c[None, :, :], axis=2), axis=1)
    return float(np.median(near))


# --------------------------------------------------------------------------- #
# Surface records (plain dicts, JSON-friendly)
# --------------------------------------------------------------------------- #
def new_surface(sid: int, n, c, up, tol: float, cell: float, uv: np.ndarray,
                point_idx: np.ndarray | None, label: str, pad_cells: int = 3) -> dict:
    """A plane with an (empty) outline grid covering the given in-plane points (u, v)."""
    n = np.asarray(n, np.float64)
    u, v = _basis(n, up)
    lo = uv.min(0) - pad_cells * cell
    hi = uv.max(0) + pad_cells * cell
    shape = (max(1, int(math.ceil((hi[1] - lo[1]) / cell))),
             max(1, int(math.ceil((hi[0] - lo[0]) / cell))))  # rows = v, cols = u
    return {"id": int(sid), "label": label, "normal": n.tolist(), "centre": list(map(float, c)),
            "u": u.tolist(), "v": v.tolist(), "tol": float(tol), "cell": float(cell),
            "origin": [float(lo[0]), float(lo[1])], "shape": list(shape),
            "mask": mask_to_str(np.zeros(shape, bool)),
            "point_idx": (point_idx.astype(np.int64).tolist() if point_idx is not None else []),
            "rule": None, "material": "", "confidence": None, "reason": "", "stats": {}}


class Geo:
    """numpy view of a surface record (cached arrays + outline helpers). Call save() after
    changing the outline."""

    def __init__(self, s: dict):
        self.s = s
        self.n = np.asarray(s["normal"], np.float64)
        self.c = np.asarray(s["centre"], np.float64)
        self.u = np.asarray(s["u"], np.float64)
        self.v = np.asarray(s["v"], np.float64)
        self.cell = float(s["cell"])
        self.o = np.asarray(s["origin"], np.float64)
        self.shape = tuple(s["shape"])
        self.mask = mask_from_str(s["mask"], self.shape)

    def save(self):
        self.shape = tuple(self.mask.shape)
        self.s["mask"] = mask_to_str(self.mask)
        self.s["origin"] = [float(self.o[0]), float(self.o[1])]
        self.s["shape"] = list(self.shape)

    def to_uv(self, X: np.ndarray) -> np.ndarray:
        d = X - self.c
        return np.stack([d @ self.u, d @ self.v], -1)

    def cell_of(self, uv: np.ndarray):
        """(row, col, inside_grid) for in-plane coordinates."""
        col = np.floor((uv[..., 0] - self.o[0]) / self.cell).astype(np.int64)
        row = np.floor((uv[..., 1] - self.o[1]) / self.cell).astype(np.int64)
        ok = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        return row, col, ok

    def inside(self, uv: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        m = self.mask if mask is None else mask
        r, c, ok = self.cell_of(uv)
        out = np.zeros(ok.shape, bool)
        out[ok] = m[r[ok], c[ok]]
        return out

    def cell_centres(self, mask: np.ndarray | None = None) -> np.ndarray:
        m = self.mask if mask is None else mask
        r, c = np.nonzero(m)
        uu = self.o[0] + (c + 0.5) * self.cell
        vv = self.o[1] + (r + 0.5) * self.cell
        return self.c + uu[:, None] * self.u + vv[:, None] * self.v

    def grow_to(self, uv: np.ndarray, pad: int = 2, max_cells: int = 4_000_000):
        """Enlarge the grid so it covers uv (keeps existing cells, stays aligned)."""
        if len(uv) == 0:
            return
        hi_old = self.o + np.array([self.shape[1], self.shape[0]]) * self.cell
        lo = np.minimum(self.o, uv.min(0) - pad * self.cell)
        hi = np.maximum(hi_old, uv.max(0) + pad * self.cell)
        shift = np.ceil((self.o - lo) / self.cell - 1e-9).astype(int)
        new_o = self.o - shift * self.cell
        shape = (int(math.ceil((hi[1] - new_o[1]) / self.cell - 1e-9)),
                 int(math.ceil((hi[0] - new_o[0]) / self.cell - 1e-9)))
        shape = (max(shape[0], shift[1] + self.shape[0]), max(shape[1], shift[0] + self.shape[1]))
        if shape[0] * shape[1] > max_cells:
            return
        if shape == tuple(self.shape) and (shift == 0).all():
            return
        m = np.zeros(shape, bool)
        m[shift[1]:shift[1] + self.shape[0], shift[0]:shift[0] + self.shape[1]] = self.mask
        self.mask, self.o, self.shape = m, new_o, shape

    def area(self) -> float:
        return float(self.mask.sum()) * self.cell ** 2


def clean_mask(m: np.ndarray, close: int = 2, max_hole: int = 30) -> np.ndarray:
    """Close small gaps and fill small enclosed holes (a doorway or window stays open)."""
    import cv2
    if not m.any():
        return m
    k = np.ones((2 * close + 1, 2 * close + 1), np.uint8)
    out = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k)
    pad = np.pad(out, 1)
    ff = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    outside = pad.copy()
    cv2.floodFill(outside, ff, (0, 0), 2)
    holes = (outside[1:-1, 1:-1] == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] <= max_hole
    return out.astype(bool) | small[lab]


# --------------------------------------------------------------------------- #
# 1. Candidates
# --------------------------------------------------------------------------- #
def find_candidates(points: np.ndarray, up: np.ndarray, cams: np.ndarray, s: dict,
                    seed: int = 0) -> tuple[list[dict], dict]:
    """Every flat patch worth a look. Returns (surfaces, info)."""
    import cv2
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, np.float64)
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    cams = np.asarray(cams, np.float64)
    n_all = len(pts)
    typ = typical_distance(pts, cams, rng)
    tol = float(s["scaffold_tol"]) * typ
    cell = float(s["scaffold_cell"]) * typ
    min_in = max(40, int(float(s["scaffold_cand_min_frac"]) * n_all))
    remaining = np.arange(n_all)
    out: list[dict] = []
    sid = 0
    for _ in range(int(s["scaffold_candidates"]) * 2):
        if len(out) >= int(s["scaffold_candidates"]) or len(remaining) < min_in:
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
            inl = np.abs((P - a) @ n) < tol
            k = int(inl.sum())
            if k > best_n:
                best, best_n = inl, k
        if best is None or best_n < min_in:
            break
        n, c = _fit_plane(P[best])
        dist = np.abs((P - c) @ n)
        inl = dist < tol
        idx_all = remaining[inl]
        remaining = remaining[~(dist < 2 * tol)]  # don't find the same surface twice
        # split into connected pieces on the plane (two coplanar walls are two surfaces)
        u, v = _basis(n, up)
        q = pts[idx_all]
        uv = np.stack([(q - c) @ u, (q - c) @ v], 1)
        lo = uv.min(0)
        shape = (int((uv[:, 1].max() - lo[1]) / cell) + 1, int((uv[:, 0].max() - lo[0]) / cell) + 1)
        if shape[0] * shape[1] > 4_000_000:
            continue
        rr = ((uv[:, 1] - lo[1]) / cell).astype(int)
        cc = ((uv[:, 0] - lo[0]) / cell).astype(int)
        occ = np.zeros(shape, np.uint8)
        occ[rr, cc] = 1
        occ_b = cv2.dilate(occ, np.ones((5, 5), np.uint8))
        ncomp, lab = cv2.connectedComponents(occ_b, connectivity=8)
        point_comp = lab[rr, cc]
        for k in range(1, ncomp):
            sel = point_comp == k
            if sel.sum() < max(30, min_in // 2):
                continue
            pidx = idx_all[sel]
            n2, c2 = _fit_plane(pts[pidx])
            if n2 @ n < 0:
                n2 = -n2
            u2, v2 = _basis(n2, up)
            uv2 = np.stack([(pts[pidx] - c2) @ u2, (pts[pidx] - c2) @ v2], 1)
            label = orientation(n2, up, cams, c2, float(s["scaffold_max_tilt_deg"]))
            srf = new_surface(sid, n2, c2, up, tol, cell, uv2, pidx, label)
            g = Geo(srf)
            r_, c_, ok = g.cell_of(uv2)
            g.mask[r_[ok], c_[ok]] = True
            g.mask = clean_mask(g.mask, close=1)
            g.save()
            d2 = np.abs((pts[pidx] - c2) @ n2)
            srf["stats"] = {"points": int(len(pidx)),
                            "concentration": round(concentration(g, pts, tol), 3),
                            "median_offset": round(float(np.median(d2) / tol), 3),
                            "points_area": round(g.area(), 3)}
            out.append(srf)
            sid += 1
    n_found = len(out)
    out = merge_stacked(out, pts, up, cams, s)
    n_joined = sum(len(x["stats"].get("merged_from", [])) for x in out)
    cap = int(s["scaffold_candidates"])
    out = sorted(out, key=lambda x: -x["stats"]["points"])[:cap]  # pieces can overshoot the cap
    for k, srf in enumerate(out):
        srf["id"] = k
    info = {"typical_distance": round(typ, 4), "tol": round(tol, 5), "cell": round(cell, 5),
            "sfm_points": n_all, "candidates": len(out), "found": n_found,
            "stacked_layers_joined": n_joined}
    log.info("Scaffold: %d candidate surface(s) from %d SfM points (plane thickness %.3g, "
             "outline cell %.3g)", len(out), n_all, tol, cell)
    return out, info


def thickness(srf: dict) -> float:
    """How thick the surface's points are: the plane tolerance, or more for a surface joined
    from stacked layers (used to measure concentration fairly)."""
    return max(float(srf["tol"]), float(srf.get("thickness") or 0))


def concentration(g: "Geo", pts: np.ndarray, tol: float, coarse: int = 4) -> float:
    """Share of the SfM points near this surface (within 4x the plane thickness) that lie ON
    it (within the thickness). ~1 for paving or a wall, ~0.25 for a slice through a bush.
    Measured per surface over its footprint on a coarse grid (coarse x the outline cell,
    plus one coarse cell around): the outline's own cells exist only where on-plane points
    are, so counting just inside them would make a sparse bush slice look flat."""
    import cv2
    if not g.mask.any():
        return 0.0
    d = pts - g.c
    off = np.abs(d @ g.n)
    band = off < 4 * tol
    if not band.any():
        return 0.0
    q = d[band]
    k = max(1, int(coarse))
    H, W = g.shape
    cm = np.zeros((-(-H // k) + 2, -(-W // k) + 2), np.uint8)
    r, c = np.nonzero(g.mask)
    cm[r // k + 1, c // k + 1] = 1
    cm = cv2.dilate(cm, np.ones((3, 3), np.uint8)).astype(bool)
    cc = np.floor(((q @ g.u) - g.o[0]) / (g.cell * k)).astype(np.int64) + 1
    rr = np.floor(((q @ g.v) - g.o[1]) / (g.cell * k)).astype(np.int64) + 1
    ok = (rr >= 0) & (rr < cm.shape[0]) & (cc >= 0) & (cc < cm.shape[1])
    near = np.zeros(len(q), bool)
    near[ok] = cm[rr[ok], cc[ok]]
    on = near & (off[band] < tol)
    return float(on.sum()) / max(int(near.sum()), 1)


def _overlap_share(A: "Geo", B: "Geo") -> float:
    """Share of the smaller outline that the other one covers (seen along the normal)."""
    na, nb = int(A.mask.sum()), int(B.mask.sum())
    if not na or not nb:
        return 0.0
    small, big = (A, B) if na <= nb else (B, A)
    cc = small.cell_centres()
    return float(big.inside(big.to_uv(cc)).sum()) / len(cc)


def _continuous(A: "Geo", B: "Geo", tol: float, reach: int = 2, min_cells: int = 5) -> bool:
    """A and B touch (within `reach` cells) and meet without a step where they touch: the
    two planes are less than the plane thickness apart along the shared edge."""
    import cv2
    grown = cv2.dilate(B.mask.astype(np.uint8),
                       np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)).astype(bool)
    cc = A.cell_centres()
    if not len(cc):
        return False
    touch = cc[B.inside(B.to_uv(cc), grown)]
    if len(touch) < min_cells:
        return False
    return float(np.median(np.abs((touch - B.c) @ B.n))) < tol


def merge_stacked(surfaces: list[dict], pts: np.ndarray, up: np.ndarray, cams: np.ndarray,
                  s: dict, max_angle_deg: float = 5.0, min_overlap: float = 0.2) -> list[dict]:
    """RANSAC with a thin plane finds a floor that is a little thicker or more tilted than
    the tolerance as several parallel layers on top of each other (their outlines then get
    shared out cell by cell: a checkerboard). Join nearly parallel candidates that are close
    together AND either overlap seen from above, or touch and meet without a step (a slightly
    tilted RANSAC plane cuts the layers into side-by-side strips). A real step's two levels
    sit side by side with a jump where they meet, so they stay apart; the joined surface must
    also still be flat."""
    if len(surfaces) < 2:
        return surfaces
    geos = [Geo(x) for x in surfaces]
    cos_max = math.cos(math.radians(max_angle_deg))
    parent = list(range(len(surfaces)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a in range(len(surfaces)):
        for b in range(a + 1, len(surfaces)):
            A, B = geos[a], geos[b]
            if surfaces[a]["label"] != surfaces[b]["label"] or abs(float(A.n @ B.n)) < cos_max:
                continue
            tol = max(float(surfaces[a]["tol"]), float(surfaces[b]["tol"]))
            if abs(float((B.c - A.c) @ A.n)) > 4 * tol:
                continue
            if _overlap_share(A, B) >= min_overlap or _continuous(A, B, tol):
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for k in range(len(surfaces)):
        groups.setdefault(root(k), []).append(k)
    out = []
    n_merged = 0
    for members in groups.values():
        if len(members) == 1:
            out.append(surfaces[members[0]])
            continue
        merged = _join(surfaces, geos, members, pts, up, cams, s)
        if merged is None:  # not flat once joined (a real step): keep them apart
            out.extend(surfaces[k] for k in members)
            continue
        out.append(merged)
        n_merged += len(members)
    if n_merged:
        log.info("Scaffold: joined %d stacked layer(s) of the same surface into %d",
                 n_merged, sum(1 for x in out if "merged_from" in x["stats"]))
    return out


def _join(surfaces, geos, members, pts, up, cams, s) -> dict | None:
    """One surface from several stacked candidates, or None if the result isn't flat."""
    tol = max(float(surfaces[k]["tol"]) for k in members)
    cell = min(float(surfaces[k]["cell"]) for k in members)
    pidx = np.unique(np.concatenate([np.asarray(surfaces[k]["point_idx"], np.int64)
                                     for k in members]))
    P = pts[pidx]
    n, c = _fit_plane(P)
    n0 = geos[members[0]].n
    if n @ n0 < 0:
        n = -n
    signed = (P - c) @ n
    off = np.abs(signed)
    # neighbouring layers were at most 4x the thickness apart; a joined surface much thicker
    # than a few layers is not one flat surface
    lo, hi = np.percentile(signed, [5, 95])
    if float(hi - lo) > 6 * tol:
        return None
    cells = np.concatenate([geos[k].cell_centres() for k in members])
    u, v = _basis(n, up)
    uv_cells = np.stack([(cells - c) @ u, (cells - c) @ v], 1)
    uv_pts = np.stack([(P - c) @ u, (P - c) @ v], 1)
    label = orientation(n, up, cams, c, float(s["scaffold_max_tilt_deg"]))
    srf = new_surface(-1, n, c, up, tol, cell, np.concatenate([uv_cells, uv_pts]), pidx, label)
    g = Geo(srf)
    for uv in (uv_cells, uv_pts):
        r_, c_, ok = g.cell_of(uv)
        g.mask[r_[ok], c_[ok]] = True
    g.mask = clean_mask(g.mask, close=1)
    g.save()
    srf["thickness"] = max(tol, float(hi - lo) / 2)  # half the joined layers' spread
    conc = concentration(g, pts, srf["thickness"])
    # stacked layers of one surface hug the joined plane; parallel slices through a bush
    # stay ~0.25 at any thickness, so they never become a fake slab
    if conc < float(s["scaffold_min_concentration"]):
        return None
    srf["stats"] = {"points": int(len(pidx)),
                    "concentration": round(conc, 3),
                    "median_offset": round(float(np.median(off) / tol), 3),
                    "points_area": round(g.area(), 3),
                    "merged_from": [int(surfaces[k]["id"]) for k in members]}
    return srf


# --------------------------------------------------------------------------- #
# Projection of surfaces into a camera
# --------------------------------------------------------------------------- #
def pixel_rays(fr: dict, step: int, W: int | None = None, H: int | None = None):
    """World ray origin and directions (camera-z = 1) for a pixel grid at W x H (default:
    the frame's size), sampled every `step` pixels; also the pixel coords."""
    K, Vm = fr["K"].astype(np.float64), fr["viewmat"].astype(np.float64)
    W = W or fr["W"]
    H = H or fr["H"]
    sx, sy = W / fr["W"], H / fr["H"]
    vv, uu = np.mgrid[step // 2:H:step, step // 2:W:step]
    dirs_c = np.stack([((uu + 0.5) / sx - K[0, 2]) / K[0, 0],
                       ((vv + 0.5) / sy - K[1, 2]) / K[1, 1],
                       np.ones_like(uu, np.float64)], -1)
    R, t = Vm[:3, :3], Vm[:3, 3]
    o = -R.T @ t
    return o, dirs_c @ R, uu, vv


def hit(g: Geo, o: np.ndarray, dirs_w: np.ndarray, use_mask: np.ndarray | None = None,
        grid_only: bool = False):
    """(camera-z depth, inside, uv) of the plane along each ray. inside = in the outline
    (or, with grid_only, anywhere in the outline's grid)."""
    denom = dirs_w @ g.n
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((g.c - o) @ g.n) / denom
    good = np.isfinite(z) & (z > 0) & (np.abs(denom) > 1e-6)
    X = o + dirs_w * np.where(good, z, 0)[..., None]
    uv = g.to_uv(X)
    if grid_only:
        _, _, ok = g.cell_of(uv)
        return z, good & ok, uv
    return z, good & g.inside(uv, use_mask), uv


def project_points(fr: dict, X: np.ndarray, W: int | None = None, H: int | None = None):
    """Pixel coords (x, y) and camera-z of world points in a frame (at W x H)."""
    W = W or fr["W"]
    H = H or fr["H"]
    K, Vm = fr["K"].astype(np.float64), fr["viewmat"].astype(np.float64)
    Xc = X @ Vm[:3, :3].T + Vm[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (K[0, 0] * Xc[:, 0] / z + K[0, 2]) * W / fr["W"]
        y = (K[1, 1] * Xc[:, 1] / z + K[1, 2]) * H / fr["H"]
    return x, y, z


# --------------------------------------------------------------------------- #
# 2. Photo check
# --------------------------------------------------------------------------- #
def sample_images(idx: list[int], n: int) -> list[int]:
    idx = list(idx)
    if len(idx) <= n:
        return idx
    return [idx[int(k)] for k in np.linspace(0, len(idx) - 1, n)]


def photo_check(surfaces: list[dict], ds, load_prior, image_idx: list[int], s: dict,
                grow: bool = True) -> dict:
    """Compare every surface with the photos' depth estimates. Adds stats to each surface,
    records each surface's best views and (grow=True) fills outlines where the photos agree."""
    import cv2
    step = int(s["scaffold_check_step"])
    agree = float(s["scaffold_agree"])
    geos = [Geo(x) for x in surfaces]
    for g in geos if grow else []:  # room to grow into areas the photos confirm (25% each way)
        span = max(g.shape) * g.cell
        lo, hi = g.o, g.o + np.array([g.shape[1], g.shape[0]]) * g.cell
        g.grow_to(np.array([lo - 0.25 * span, hi + 0.25 * span]), pad=0)
    names = ("seen", "agree", "behind", "front", "views")
    acc = [{k: np.zeros(g.shape, np.float32) for k in names} for g in geos]
    views: list[list] = [[] for _ in geos]
    n_used = 0
    for i in image_idx:
        p = load_prior(i)
        if p is None:
            continue
        depth = p[0]
        fr = ds.frames[i]
        o, dirs, uu, vv = pixel_rays(fr, step)
        if depth.shape != (fr["H"], fr["W"]):
            depth = cv2.resize(depth, (fr["W"], fr["H"]), interpolation=cv2.INTER_NEAREST)
        pri = depth[np.clip(vv, 0, fr["H"] - 1), np.clip(uu, 0, fr["W"] - 1)]
        valid = pri > 0
        n_used += 1
        for k, g in enumerate(geos):
            z, ing, uv = hit(g, o, dirs, grid_only=True)
            m = ing & valid
            if not m.any():
                continue
            rel = (pri - z) / np.maximum(z, 1e-9)
            a = m & (np.abs(rel) < agree)
            b = m & (rel > agree)    # the photo sees further: through it, or it isn't there
            f = m & (rel < -agree)   # something in front of it
            r_, c_, ok = g.cell_of(uv)
            for name, sel in (("seen", m), ("agree", a), ("behind", b), ("front", f)):
                ss = sel & ok
                np.add.at(acc[k][name], (r_[ss], c_[ss]), 1)
            if a.any():
                vm = np.zeros(g.shape, bool)
                vm[r_[a & ok], c_[a & ok]] = True
                acc[k]["views"] += vm
            inside = g.inside(uv) & m
            views[k].append((int(inside.sum()), int((a & inside).sum()), i))
    min_views = int(s["scaffold_min_views"])
    for k, (srf, g) in enumerate(zip(surfaces, geos)):
        A = acc[k]
        base = g.mask.copy()
        if grow:
            photo = (A["views"] >= 2) & (A["agree"] >= 0.5 * np.maximum(A["seen"], 1))
            g.mask = clean_mask(base | photo, close=2)
            g.save()
        srf["stats"]["views"] = int(sum(1 for t in views[k] if t[1] > 20))
        srf["stats"]["enough_views"] = srf["stats"]["views"] >= min_views
        srf["stats"]["photo_filled_share"] = round(
            float((g.mask & ~base).sum()) / max(int(g.mask.sum()), 1), 3)
        srf["_acc"] = A  # per-cell photo evidence (in memory only), for resolve_overlaps
        _photo_stats(srf, g, A)
        views[k].sort(key=lambda t: -(t[1] + 0.2 * t[0]))
        srf["best_views"] = [t[2] for t in views[k][:6] if t[0] > 0]
    if grow:
        log.info("Scaffold: checked %d candidate(s) against %d photo(s)", len(surfaces), n_used)
    return {"images_checked": n_used}


def _photo_stats(srf: dict, g: Geo, A: dict):
    sel = g.mask
    seen = float(A["seen"][sel].sum())
    srf["stats"].update({
        "area": round(g.area(), 3),
        "photo_agree": round(float(A["agree"][sel].sum()) / max(seen, 1), 3),
        "see_through": round(float(A["behind"][sel].sum()) / max(seen, 1), 3),
        "hidden": round(float(A["front"][sel].sum()) / max(seen, 1), 3)})


def resolve_overlaps(surfaces: list[dict], s: dict, max_angle_deg: float = 5.0) -> int:
    """Nearly parallel surfaces that overlap (e.g. a floor found as several pieces at
    slightly different heights): each overlapping patch goes to the surface the photos
    agree with there, so no area has two competing planes. Planes themselves don't move,
    so a real step between two levels stays a step. Returns how many cells moved."""
    moved = 0
    geos = [Geo(x) for x in surfaces]
    cos_max = math.cos(math.radians(max_angle_deg))
    order = sorted(range(len(surfaces)), key=lambda k: -surfaces[k]["stats"].get("photo_agree", 0))
    for ai in order:
        for bi in order:
            if ai == bi:
                continue
            A, B = geos[ai], geos[bi]
            sa, sb = surfaces[ai], surfaces[bi]
            if "_acc" not in sa or "_acc" not in sb or abs(float(A.n @ B.n)) < cos_max:
                continue
            gap = abs(float((B.c - A.c) @ A.n))
            if gap > 4 * max(float(sa["tol"]), float(sb["tol"])):
                continue
            ra, ca = np.nonzero(A.mask)
            if len(ra) == 0:
                continue
            cc = A.cell_centres()
            rb, cb, ok = B.cell_of(B.to_uv(cc))  # A's cells as seen on B's plane
            both = ok.copy()
            both[ok] = B.mask[rb[ok], cb[ok]]
            if not both.any():
                continue
            Aa, Ab = sa["_acc"], sb["_acc"]
            score_a = Aa["agree"][ra[both], ca[both]] / (Aa["seen"][ra[both], ca[both]] + 1)
            score_b = Ab["agree"][rb[both], cb[both]] / (Ab["seen"][rb[both], cb[both]] + 1)
            b_wins = score_b > score_a
            idx = np.nonzero(both)[0]
            lose_a, lose_b = idx[b_wins], idx[~b_wins]
            A.mask[ra[lose_a], ca[lose_a]] = False
            B.mask[rb[lose_b], cb[lose_b]] = False
            moved += len(idx)
    for srf, g in zip(surfaces, geos):
        g.save()
        if "_acc" in srf:
            _photo_stats(srf, g, srf["_acc"])
    if moved:
        log.info("Scaffold: split %d overlapping outline cell(s) between nearly parallel "
                 "surfaces (each goes to the one the photos agree with)", moved)
    return moved


# --------------------------------------------------------------------------- #
# 3. Automatic rules (no AI)
# --------------------------------------------------------------------------- #
def min_area(srf: dict, s: dict) -> float:
    """scaffold_min_area x (typical camera-to-surface distance)^2."""
    typ = srf["tol"] / max(float(s["scaffold_tol"]), 1e-9)
    return float(s["scaffold_min_area"]) * typ * typ


def auto_rule(srf: dict, s: dict) -> tuple[str, str]:
    st = srf["stats"]
    if not st.get("enough_views"):
        return "ignore", "seen from too few viewpoints"
    if st.get("area", 0) < min_area(srf, s):
        return "ignore", "too small"
    if st.get("see_through", 0) > 0.15:
        return "ignore", (f"photos often see through it ({st['see_through']:.2f}): glass, an "
                          "opening, or a slice through grass/planting")
    flat_pts = (st.get("concentration", 0) >= float(s["scaffold_min_concentration"])
                and st.get("points", 0) >= 200)
    flat_photo = st.get("photo_agree", 0) >= float(s["scaffold_photo_min"])
    if flat_pts or flat_photo:
        why = []
        if flat_pts:
            why.append(f"points hug the plane ({st['concentration']:.2f})")
        if flat_photo:
            why.append(f"photos agree ({st['photo_agree']:.2f})")
        return "solid", " and ".join(why)
    return "ignore", (f"not clearly flat (points {st.get('concentration', 0):.2f}, "
                      f"photos {st.get('photo_agree', 0):.2f})")


def auto_rules(surfaces: list[dict], s: dict, only_unset: bool = False):
    """Record the automatic suggestion on every surface; apply it (to all, or only to the
    ones nobody decided)."""
    for srf in surfaces:
        r, why = auto_rule(srf, s)
        srf["auto_rule"], srf["auto_reason"] = r, why
        if only_unset and srf.get("rule") in RULES:
            continue
        srf["rule"], srf["reason"], srf["decided_by"] = r, f"automatic: {why}", "auto"


# --------------------------------------------------------------------------- #
# Decision files
# --------------------------------------------------------------------------- #
def fingerprint(points: np.ndarray, alignment) -> str:
    p = np.asarray(points, np.float64)
    a = np.asarray(alignment if alignment is not None else np.eye(4), np.float64)
    vals = [len(p), *np.round(p.mean(0), 3), *np.round(p.std(0), 3), *np.round(a.ravel(), 4)]
    return ",".join(f"{v:g}" for v in vals)


def save_decision(path: Path, surfaces: list[dict], meta: dict):
    keep = [{k: v for k, v in x.items() if not k.startswith("_") and k != "point_idx"}
            for x in surfaces]
    Path(path).write_text(json.dumps({"version": VERSION, **meta, "surfaces": keep}, indent=1))


def load_decision(path: Path, fp: str) -> tuple[list[dict] | None, str]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return None, f"unreadable ({e})"
    if d.get("version") != VERSION:
        return None, f"made by a different scaffold version ({d.get('version')})"
    if d.get("fingerprint") != fp:
        return None, "made for a different reconstruction"
    for x in d["surfaces"]:
        if x.get("rule") not in RULES:
            return None, f"surface {x.get('id')} has an unknown rule {x.get('rule')!r}"
        Geo(x)  # raises on a broken outline
    return d["surfaces"], "ok"


# --------------------------------------------------------------------------- #
# 4. Runtime scaffold (used by the trainer)
# --------------------------------------------------------------------------- #
class Scaffold:
    """Surfaces with a rule other than ignore. Per image: the nearest surface along every
    pixel ray (computed at 1/step resolution and cached)."""

    RULE_CODE = {"solid": 0, "glass": 1, "soft": 2, "window": 3}

    def __init__(self, surfaces: list[dict], s: dict):
        self.surfaces = [x for x in surfaces if x.get("rule") in self.RULE_CODE]
        self.geos = [Geo(x) for x in self.surfaces]
        self.rules = [x["rule"] for x in self.surfaces]
        self.step = max(1, int(s["scaffold_res_step"]))
        self.agree = float(s["scaffold_agree"])
        self.margin = float(s["scaffold_margin"])
        self.soft_margin = float(s["scaffold_soft_margin"])
        self._rule_of = np.array([self.RULE_CODE[r] for r in self.rules] + [255], np.uint8)
        self._cache: dict[str, tuple] = {}

    def __len__(self):
        return len(self.surfaces)

    def depth(self, fr: dict):
        """(depth float16, surface index uint8 (255 = none)) at low resolution."""
        key = fr["name"]
        if key in self._cache:
            return self._cache[key]
        o, dirs, _, _ = pixel_rays(fr, self.step)
        best = np.full(dirs.shape[:2], np.inf)
        ids = np.full(dirs.shape[:2], 255, np.uint8)
        order = sorted(range(min(len(self.geos), 255)),
                       key=lambda k: self.rules[k] in SHEET_RULES)
        for k in order:  # glass/window last, so it wins ties with the wall it sits in
            g = self.geos[k]
            z, inside, _ = hit(g, o, dirs)
            slack = 2 * float(g.s["tol"]) if self.rules[k] in SHEET_RULES else 0.0
            closer = inside & (z < best + slack)
            best = np.where(closer, z, best)
            ids[closer] = k
        out = (np.where(np.isfinite(best), best, 0).astype(np.float16), ids)
        self._cache[key] = out
        return out

    def merge_prior(self, fr: dict, depth: np.ndarray, weight: np.ndarray, conf: float):
        """(depth, weight, limit, ids): the prior with exact surface depth where a solid or
        glass surface is visible; `limit` = the depth nothing may be rendered in front of
        (0 = no rule); `ids` = which surface owns each pixel (255 = none). limit and ids
        are None when no surface is visible in this image. A window only sets `limit`:
        what is seen through it keeps the depth estimate's depth."""
        import cv2
        H, W = depth.shape
        low, lid = self.depth(fr)
        plane = cv2.resize(low.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        ids = cv2.resize(lid, (W, H), interpolation=cv2.INTER_NEAREST)
        has = (ids != 255) & (plane > 0)
        if not has.any():
            return depth, weight, None, None
        rk = self._rule_of[np.where(has, ids, len(self.rules)).astype(np.int64)]
        valid = depth > 0
        rel = np.where(valid, (depth - plane) / np.maximum(plane, 1e-9), 0)
        solid_vis = has & (rk == 0) & valid & (np.abs(rel) < self.agree)
        # glass: from the outline; only "something clearly in front of the glass" disqualifies
        sheet = ~valid | (rel > -self.agree)
        glass_vis = has & (rk == 1) & sheet
        window_vis = has & (rk == 3) & sheet
        soft_vis = has & (rk == 2) & valid & (np.abs(rel) < self.soft_margin)
        exact = solid_vis | glass_vis
        vis = exact | soft_vis | window_vis
        if not vis.any():
            return depth, weight, None, None
        d2 = np.where(exact, plane, depth).astype(np.float32)
        w2 = np.where(exact, np.maximum(weight, conf), weight).astype(np.float32)
        margin = np.where(soft_vis, self.soft_margin, self.margin)
        limit = np.where(vis, plane * (1 - margin), 0).astype(np.float32)
        out_ids = np.where(vis, ids, 255).astype(np.uint8)
        return d2, w2, limit, out_ids


def free_space_loss(rend_depth, limit):
    """Penalise anything rendered in front of the depth limit (empty space)."""
    import torch
    valid = limit > 0
    if int(valid.sum()) < 64:
        return rend_depth.sum() * 0.0
    lim = limit[valid]
    return (torch.relu(lim - rend_depth[valid]) / lim).mean()


class FlatTerm:
    """Splats lying on a solid or glass surface are pushed to be thin along its normal
    (not windows: the room seen through one isn't flat)."""

    def __init__(self, scaffold: Scaffold, device):
        import torch
        self.items = []
        for g, r in zip(scaffold.geos, scaffold.rules):
            if r not in ("solid", "glass"):
                continue
            f32 = dict(dtype=torch.float32, device=device)
            self.items.append({
                "n": torch.tensor(g.n, **f32), "c": torch.tensor(g.c, **f32),
                "u": torch.tensor(g.u, **f32), "v": torch.tensor(g.v, **f32),
                "o": torch.tensor(g.o, **f32), "cell": float(g.cell), "tol": float(g.s["tol"]),
                "mask": torch.from_numpy(g.mask).to(device)})
        if self.items:
            self.normals = torch.stack([it["n"] for it in self.items])
            self.tols = torch.tensor([it["tol"] for it in self.items], device=device)
        self.members = None
        self.n_at = -1

    def __bool__(self):
        return bool(self.items)

    def refresh(self, means) -> int:
        import torch
        with torch.no_grad():
            N = len(means)
            owner = torch.full((N,), -1, dtype=torch.long, device=means.device)
            for k, it in enumerate(self.items):
                d = means - it["c"]
                near = ((d @ it["n"]).abs() < 2 * it["tol"]) & (owner < 0)
                if not near.any():
                    continue
                idx = near.nonzero().squeeze(1)
                col = (((d[idx] @ it["u"]) - it["o"][0]) / it["cell"]).floor().long()
                row = (((d[idx] @ it["v"]) - it["o"][1]) / it["cell"]).floor().long()
                Hm, Wm = it["mask"].shape
                ok = (row >= 0) & (row < Hm) & (col >= 0) & (col < Wm)
                ins = torch.zeros_like(ok)
                ins[ok] = it["mask"][row[ok], col[ok]]
                owner[idx[ins]] = k
            sel = (owner >= 0).nonzero().squeeze(1)
            self.members = (sel, owner[sel])
            self.n_at = N
            return int(len(sel))

    def loss(self, quats, log_scales):
        import torch
        import torch.nn.functional as F
        sel, own = self.members
        if len(sel) == 0:
            return log_scales.sum() * 0.0
        normals, tols = self.normals[own], self.tols[own]
        q = F.normalize(quats[sel], dim=-1)
        w, x, y, z = q.unbind(-1)
        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
        sc = torch.exp(log_scales[sel])
        proj = (R * normals[:, :, None]).sum(1)  # n . (each local axis)
        std_n = torch.sqrt((proj * proj * sc * sc).sum(-1) + 1e-12)
        return (std_n / tols).clamp(max=10).mean()


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
           (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]


def colour_for(srf: dict, by_rule: bool):
    """BGR."""
    if by_rule:
        return RULE_COLOURS.get(srf.get("rule") or "ignore", (150, 150, 150))
    r, g, b = PALETTE[srf["id"] % len(PALETTE)]
    return (b, g, r)


def _label(img, txt, x, y, scale=0.55):
    import cv2
    cv2.putText(img, txt, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


def draw_topdown(surfaces: list[dict], points: np.ndarray, cams: np.ndarray, up: np.ndarray,
                 by_rule: bool, size: int = 1000, title: str = "") -> np.ndarray:
    """Top-down map (BGR): SfM points grey, cameras red, surfaces filled + numbered."""
    import cv2
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    a, _ = _basis(up)
    b = np.cross(up, a)
    P2 = np.stack([points @ a, points @ b], 1)
    C2 = np.stack([cams @ a, cams @ b], 1)
    lo, hi = np.percentile(P2, 1, axis=0), np.percentile(P2, 99, axis=0)
    lo, hi = np.minimum(lo, C2.min(0)), np.maximum(hi, C2.max(0))
    S = size
    sc = (S - 60) / max(float(max(hi - lo)), 1e-9)

    def to_px(q):
        return ((q - lo) * sc + 30).astype(int)

    img = np.full((S, S, 3), 255, np.uint8)
    for x, y in to_px(P2[:: max(1, len(P2) // 80000)]):
        if 0 <= x < S and 0 <= y < S:
            img[y, x] = (175, 175, 175)
    over = img.copy()
    shown = [x for x in surfaces if not (by_rule and x.get("rule") in (None, "ignore"))]
    for srf in sorted(shown, key=lambda t: -t["stats"].get("area", 0)):
        g = Geo(srf)
        cc = g.cell_centres()
        if len(cc) == 0:
            continue
        q = to_px(np.stack([cc @ a, cc @ b], 1))
        col = colour_for(srf, by_rule)
        rad = max(1, int(g.cell * sc * 0.6))
        for x, y in q[:: max(1, len(q) // 20000)]:
            cv2.rectangle(over, (int(x) - rad, int(y) - rad), (int(x) + rad, int(y) + rad), col, -1)
    img = cv2.addWeighted(over, 0.6, img, 0.4, 0)
    for x, y in to_px(C2):
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 220), -1)
    for srf in shown:
        cc = Geo(srf).cell_centres()
        if len(cc) == 0:
            continue
        m = cc.mean(0)
        x, y = to_px(np.array([[m @ a, m @ b]]))[0]
        _label(img, f"#{srf['id']}" + (f" {srf['rule']}" if by_rule else ""), int(x), int(y))
    cv2.putText(img, f"{title} top-down (grey: SfM points, red: camera path)".strip(),
                (10, S - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if by_rule:
        x = 10
        for k, col in RULE_COLOURS.items():
            if k == "ignore":
                continue
            cv2.rectangle(img, (x, 10), (x + 14, 24), col, -1)
            cv2.putText(img, k, (x + 18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
            x += 80
    return img


def draw_on_photo(img_rgb: np.ndarray, fr: dict, surfaces: list[dict], by_rule: bool,
                  prior: np.ndarray | None = None, alpha: float = 0.45,
                  ids_only: list[int] | None = None, show_hidden: bool = False) -> np.ndarray:
    """Photo (RGB, any size) with surfaces tinted, outlined and numbered. Parts the depth
    estimate shows are hidden behind something are left untinted (except glass/windows), or, with
    show_hidden, drawn as stripes."""
    import cv2
    H, W = img_rgb.shape[:2]
    step = max(1, int(round(max(W, H) / 256)))
    o, dirs, uu, vv = pixel_rays(fr, step, W=W, H=H)
    best = np.full(dirs.shape[:2], np.inf)
    owner = np.full(dirs.shape[:2], -1, np.int32)
    for k in sorted(range(len(surfaces)), key=lambda k: surfaces[k].get("rule") in SHEET_RULES):
        srf = surfaces[k]
        if ids_only is not None and srf["id"] not in ids_only:
            continue
        if by_rule and srf.get("rule") in (None, "ignore"):
            continue
        z, inside, _ = hit(Geo(srf), o, dirs)
        slack = 2 * float(srf["tol"]) if srf.get("rule") in SHEET_RULES else 0.0
        closer = inside & (z < best + slack)
        best = np.where(closer, z, best)
        owner[closer] = k
    hidden = np.zeros(owner.shape, bool)
    if prior is not None:
        d = cv2.resize(prior, (W, H), interpolation=cv2.INTER_NEAREST)
        pri = d[np.clip(vv, 0, H - 1), np.clip(uu, 0, W - 1)]
        hidden = (pri > 0) & np.isfinite(best) & (pri < best * 0.92) & (owner >= 0)
        for k, srf in enumerate(surfaces):
            if srf.get("rule") in SHEET_RULES:
                hidden &= owner != k
    vis_owner = np.where(hidden, -1, owner)
    out = img_rgb.copy()
    small = np.zeros(owner.shape + (3,), np.uint8)
    for k in np.unique(owner[owner >= 0]):
        small[owner == k] = colour_for(surfaces[k], by_rule)[::-1]  # BGR -> RGB
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    vm = cv2.resize((vis_owner >= 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    out[vm] = (out[vm] * (1 - alpha) + big[vm] * alpha).astype(np.uint8)
    if show_hidden and hidden.any():
        hm = cv2.resize(hidden.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        yy, xx = np.mgrid[0:H, 0:W]
        stripes = hm & (((xx + yy) // max(3, W // 64)) % 2 == 0)
        out[stripes] = (out[stripes] * 0.4 + big[stripes] * 0.6).astype(np.uint8)
    lab_owner = owner if show_hidden else vis_owner
    for k in np.unique(lab_owner[lab_owner >= 0]):
        mk = cv2.resize((lab_owner == k).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colour_for(surfaces[k], by_rule)[::-1], 2)
        ys, xs = np.nonzero(mk)
        if len(xs) < 30:
            continue
        _label(out, f"#{surfaces[k]['id']}" + (f" {surfaces[k]['rule']}" if by_rule else ""),
               int(np.median(xs)), int(np.median(ys)), 0.6)
    return out


def save_report(surfaces: list[dict], points: np.ndarray, cams: np.ndarray, up: np.ndarray,
                out_dir: Path, meta: dict):
    """scaffold.json (summary) and scaffold_topdown.jpg (final rules from above)."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = [{k: x.get(k) for k in ("id", "label", "rule", "material", "confidence",
                                      "reason", "decided_by", "auto_rule", "stats", "usage")}
               for x in surfaces]
    (out_dir / "scaffold.json").write_text(json.dumps({**meta, "surfaces": summary}, indent=2))
    try:
        cv2.imwrite(str(out_dir / "scaffold_topdown.jpg"),
                    draw_topdown(surfaces, points, cams, up, by_rule=True))
    except Exception as e:  # noqa: BLE001  (a picture is a nice-to-have)
        log.warning("Couldn't draw the scaffold picture: %s", e)
