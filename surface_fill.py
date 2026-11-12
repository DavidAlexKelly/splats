"""
Fill in the hidden parts of the scene's big flat surfaces (wall behind a bush, paving under
a planter), from what the same surface looks like where the photos do show it.

The scaffold (scaffold.py, saved with every result as scene/scaffold_decision.json) already
knows the solid surfaces exactly: a plane and an outline each. For every solid surface:

  1. texture: lay a fine grid of texels over its outline and project the real photos onto
     it. A texel counts as seen in a photo only when the splat's rendered depth there agrees
     with the plane (so a bush in front hides it) and the photo isn't at a grazing angle.
     Seen texels get the weighted average colour. Unseen texels are filled only if photos
     found something IN FRONT of them (hidden behind a bush), not if they saw PAST them (an
     opening, or a plane a little off the real surface) or never looked.
  2. fill: split the seen texture into broad shading (light falling across the wall) and
     detail (bricks, joints, slabs). Shading is smoothly extended into the hole. Detail is
     copied along the texture's own repeat (found by autocorrelation: the brick or slab
     spacing), so bricks continue as bricks with their joints lined up. Where no clear
     repeat exists, the hole gets the smooth shading only.
  3. bake: one thin, flat, opaque Gaussian per hidden texel (on a coarser grid if there are
     very many), lying on the plane, with the filled colour.

No AI model, so no invented objects. It is one flat image per surface, so it can't
disagree with itself from different viewpoints. The new Gaussians sit behind whatever hid
them, so from the flight path they stay hidden; they appear only when a viewer looks
behind the bush. The caller (fixer_refine.py) checks held-out real photos and undoes the
fill if it costs more than max_round_drop.

Previews: surface_fill/<id>_<material>.jpg = seen texture (hidden part dark red) | filled.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

import scaffold as sc

log = logging.getLogger("fixer")

C0 = 0.28209479177387814  # SH band-0 constant: colour = 0.5 + C0 * sh0

DEFAULTS = {
    "surface_fill": True,
    "surface_fill_rules": ["solid"],   # which scaffold rules get filled
    "surface_fill_photos": 160,        # training photos projected onto the surfaces
    "surface_fill_max_texels": 1_500_000,  # per surface (texel size grows to fit)
    "surface_fill_max_new": 400_000,   # new Gaussians in all (coarser where more)
    "surface_fill_min_hidden": 200,    # surfaces with fewer hidden texels are skipped
    "surface_fill_agree": 0.05,        # rendered depth within 5% of the plane = surface seen
    "surface_fill_min_cos": 0.15,      # skip photos at a grazing angle to the surface
    "surface_fill_opacity": 0.95,
}


def setting(d: dict, k: str):
    return d.get(k, DEFAULTS[k])


def load_surfaces(scene_dir: Path, ds, rules) -> tuple[list[dict], str]:
    """The scaffold's surfaces with the given rules, if the saved decision belongs to this
    reconstruction (same points and alignment)."""
    p = Path(scene_dir) / "scaffold_decision.json"
    if not p.exists():
        return [], "no scaffold_decision.json in the scene bundle (result made before the scaffold)"
    surfaces, why = sc.load_decision(p, sc.fingerprint(ds.points, ds.alignment))
    if surfaces is None:
        return [], f"saved scaffold not usable: {why}"
    keep = [x for x in surfaces if x.get("rule") in set(rules)]
    return keep, "ok" if keep else f"no surfaces with rule {', '.join(rules)}"


# --------------------------------------------------------------------------- #
# 1. Textures from the photos
# --------------------------------------------------------------------------- #
def make_texture(srf: dict, footprint: float, max_texels: int) -> dict | None:
    """A texel grid over the surface's outline: texel size ~ one photo pixel on it, coarser
    if the surface is big."""
    g = sc.Geo(srf)
    if not g.mask.any():
        return None
    rows, cols = np.nonzero(g.mask)
    lo = g.o + np.array([cols.min(), rows.min()]) * g.cell
    hi = g.o + np.array([cols.max() + 1, rows.max() + 1]) * g.cell
    span = hi - lo
    texel = max(footprint, math.sqrt(span[0] * span[1] / max(max_texels, 1)))
    W, H = max(2, int(math.ceil(span[0] / texel))), max(2, int(math.ceil(span[1] / texel)))
    uu = lo[0] + (np.arange(W) + 0.5) * texel
    vv = lo[1] + (np.arange(H) + 0.5) * texel
    U, V = np.meshgrid(uu, vv)
    uv = np.stack([U, V], -1).reshape(-1, 2)
    inside = g.inside(uv).reshape(H, W)
    X = g.c + uv[:, :1] * g.u + uv[:, 1:] * g.v
    return {"srf": srf, "geo": g, "texel": texel, "H": H, "W": W, "inside": inside,
            "X": X.astype(np.float32), "col": np.zeros((H * W, 3), np.float64),
            "w": np.zeros(H * W, np.float64)}


def observe(texs: list[dict], ds, photo_idx: list[int], view, d: dict, device):
    """Project the photos onto every texture (see the module notes)."""
    import torch
    agree, min_cos = float(setting(d, "surface_fill_agree")), float(setting(d, "surface_fill_min_cos"))
    T = [{"X": torch.from_numpy(t["X"]).to(device),
          "n": torch.as_tensor(t["geo"].n, dtype=torch.float32, device=device),
          "col": torch.zeros(len(t["X"]), 3, device=device),
          "w": torch.zeros(len(t["X"]), device=device),
          "occ": torch.zeros(len(t["X"]), device=device),      # something in front of it
          "beyond": torch.zeros(len(t["X"]), device=device)}   # the photo sees past it
         for t in texs]
    with torch.no_grad():
        for i in photo_idx:
            fr = ds.frames[i]
            img, _ = ds.load(i)
            I = torch.from_numpy(img).to(device).float().div_(255.0)
            D = view.depth(fr, 0.25)
            V = torch.as_tensor(np.asarray(fr["viewmat"]), dtype=torch.float32, device=device)
            K = torch.as_tensor(np.asarray(fr["K"]), dtype=torch.float32, device=device)
            c = torch.as_tensor(np.asarray(fr["center"], np.float32), device=device)
            Hh, Ww = img.shape[:2]
            h, w = D.shape
            for t in T:
                Xc = t["X"] @ V[:3, :3].T + V[:3, 3]
                z = Xc[:, 2]
                zc = z.clamp(min=1e-6)
                u = K[0, 0] * Xc[:, 0] / zc + K[0, 2]
                v = K[1, 1] * Xc[:, 1] / zc + K[1, 2]
                ok = (z > 0.05) & (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
                if not ok.any():
                    continue
                ui = u.long().clamp(0, Ww - 1)
                vi = v.long().clamp(0, Hh - 1)
                dd = D[(v * (h / Hh)).long().clamp(0, h - 1), (u * (w / Ww)).long().clamp(0, w - 1)]
                ray = torch.nn.functional.normalize(c[None] - t["X"], dim=1)
                cos = (ray @ t["n"]).abs()
                ok &= cos > min_cos
                seen = ok & (dd > 0) & ((dd - z).abs() < agree * z)
                wt = torch.where(seen, cos / zc.pow(2), torch.zeros_like(z))
                t["col"] += wt[:, None] * I[vi, ui]
                t["w"] += wt
                t["occ"] += (ok & (dd > 0) & (dd < (1 - agree) * z)).float()
                t["beyond"] += (ok & ((dd <= 0) | (dd > (1 + agree) * z))).float()
    for t, tt in zip(texs, T):
        for k in ("col", "w", "occ", "beyond"):
            t[k] = tt[k].cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# 2. Filling the hidden part
# --------------------------------------------------------------------------- #
def push_pull(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fill where w == 0 smoothly from the surroundings: average down a pyramid (weighted),
    then blend back up, each level keeping its own values where it has data."""
    import cv2
    levels = [(img * w[..., None], w)]
    while min(levels[-1][1].shape) > 4:
        a, b = levels[-1]
        h, ww = max(1, a.shape[0] // 2), max(1, a.shape[1] // 2)
        levels.append((cv2.resize(a.astype(np.float32), (ww, h), interpolation=cv2.INTER_AREA),
                       cv2.resize(b.astype(np.float32), (ww, h), interpolation=cv2.INTER_AREA)))
    a, b = levels[-1]
    mean = (a.reshape(-1, a.shape[-1]).sum(0) / max(float(b.sum()), 1e-9))
    cur = np.where(b[..., None] > 1e-6, a / np.maximum(b, 1e-9)[..., None], mean)
    for a, b in reversed(levels[:-1]):
        up = cv2.resize(cur.astype(np.float32), (b.shape[1], b.shape[0]),
                        interpolation=cv2.INTER_LINEAR)
        own = a / np.maximum(b, 1e-9)[..., None]
        alpha = np.clip(b * 2, 0, 1)[..., None]
        cur = alpha * own + (1 - alpha) * up
    return cur


def shading(tex: np.ndarray, seen: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Broad light across the surface, defined everywhere: a blur of the seen texels,
    extended smoothly into the hole."""
    import cv2
    k = int(sigma * 3) | 1
    num = cv2.GaussianBlur((tex * seen[..., None]).astype(np.float32), (k, k), sigma)
    den = cv2.GaussianBlur(seen.astype(np.float32), (k, k), sigma)
    have = den > 0.3
    base = np.where(have[..., None], num / np.maximum(den, 1e-6)[..., None], 0)
    return push_pull(base, have.astype(np.float32)).astype(np.float64)


def repeat_vectors(detail: np.ndarray, seen: np.ndarray, min_shift: int = 3,
                   max_frac: float = 0.45) -> tuple[list[np.ndarray], float]:
    """The texture's repeat (brick/slab spacing) as up to two shift vectors (dx, dy) in
    texels, from the normalised autocorrelation of the seen detail; and how strong it is."""
    g = detail.mean(-1) * seen
    m = seen.astype(np.float64)
    H, W = g.shape
    P = (2 * H, 2 * W)
    F = np.fft.rfft2(g, P)
    Fm = np.fft.rfft2(m, P)
    Fg2 = np.fft.rfft2(g * g, P)
    num = np.fft.irfft2(F * np.conj(F), P)
    overlap = np.fft.irfft2(Fm * np.conj(Fm), P)
    e1 = np.fft.irfft2(Fg2 * np.conj(Fm), P)  # sum of g^2 over the overlap, shifted side
    e2 = np.fft.irfft2(Fm * np.conj(Fg2), P)  # ... and the other side
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / np.sqrt(np.maximum(e1, 1e-12) * np.maximum(e2, 1e-12))
    corr[overlap < 0.2 * m.sum()] = -1
    corr = np.fft.fftshift(corr)
    cy, cx = P[0] // 2, P[1] // 2
    yy, xx = np.mgrid[0:P[0], 0:P[1]]
    dy, dx = yy - cy, xx - cx
    valid = ((np.abs(dx) >= min_shift) | (np.abs(dy) >= min_shift)) \
        & (np.abs(dx) <= max_frac * W) & (np.abs(dy) <= max_frac * H)
    corr = np.where(valid, corr, -1)
    # prefer the shortest repeat: multiples of it correlate almost as well, but copying from
    # far away fails on big holes and drifts with perspective
    corr = corr - 0.15 * np.hypot(dx / W, dy / H)
    vecs, strength = [], 0.0
    k = int(np.argmax(corr))
    if corr.flat[k] < 0.3:
        return vecs, strength
    a = np.array([dx.flat[k], dy.flat[k]])
    vecs.append(a)
    strength = float(corr.flat[k])
    # second repeat: best peak not along the first (a running bond's offset course)
    along = np.abs(dx * a[1] - dy * a[0]) < 1.5 * np.linalg.norm(a)
    corr2 = np.where(along, -1, corr)
    k = int(np.argmax(corr2))
    if corr2.flat[k] >= 0.3:
        vecs.append(np.array([dx.flat[k], dy.flat[k]]))
    return vecs, strength


def fill_texture(tex: np.ndarray, seen: np.ndarray, todo: np.ndarray, reach: int = 8):
    """tex filled at `todo` texels: shading + detail copied along the repeat. Returns
    (filled texture, repeat vectors, repeat strength, share of todo that got detail)."""
    L = shading(tex, seen)
    detail = np.where(seen[..., None], tex - L, 0)
    vecs, strength = repeat_vectors(detail, seen)
    D = np.zeros_like(detail)
    got = np.zeros(seen.shape, bool)
    H, W = seen.shape
    if vecs:
        a = vecs[0]
        b = vecs[1] if len(vecs) > 1 else np.array([0, 0])
        steps = sorted({(i, j) for i in range(-reach, reach + 1)
                        for j in (range(-reach, reach + 1) if len(vecs) > 1 else [0])
                        if (i, j) != (0, 0)},
                       key=lambda ij: np.linalg.norm(ij[0] * a + ij[1] * b))
        ys, xs = np.nonzero(todo)
        for i, j in steps:
            left = ~got[ys, xs]
            if not left.any():
                break
            sx = xs[left] + int(i * a[0] + j * b[0])
            sy = ys[left] + int(i * a[1] + j * b[1])
            ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
            ok[ok] = seen[sy[ok], sx[ok]]
            ty, tx = ys[left][ok], xs[left][ok]
            D[ty, tx] = detail[sy[ok], sx[ok]]
            got[ty, tx] = True
    out = np.where(todo[..., None], np.clip(L + D, 0, 1), tex)
    share = float(got[todo].mean()) if todo.any() else 0.0
    return out, vecs, strength, share


# --------------------------------------------------------------------------- #
# 3. Baking into Gaussians
# --------------------------------------------------------------------------- #
def _quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = local x, y, z in world) -> quaternion (w, x, y, z)."""
    m = Rm
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def bake(t: dict, filled: np.ndarray, todo: np.ndarray, stride: int, opacity: float,
         n_sh_rest: int) -> dict:
    """Flat Gaussians for the todo texels (every `stride`-th in each direction)."""
    g, texel = t["geo"], t["texel"]
    sel = np.zeros_like(todo)
    sel[::stride, ::stride] = True
    sel &= todo
    idx = np.nonzero(sel.ravel())[0]
    k = len(idx)
    rgb = filled.reshape(-1, 3)[idx]
    q = _quat_wxyz(np.stack([g.u, g.v, g.n], 1))
    size = texel * stride * 0.6
    thin = max(float(g.s["tol"]) * 0.02, size * 0.01)
    return {"means": t["X"][idx].astype(np.float32),
            "quats": np.tile(q.astype(np.float32), (k, 1)),
            "scales": np.tile(np.log(np.array([size, size, thin], np.float32)), (k, 1)),
            "opacities": np.full(k, math.log(opacity / (1 - opacity)), np.float32),
            "sh0": ((rgb - 0.5) / C0).astype(np.float32).reshape(k, 1, 3),
            "shN": np.zeros((k, n_sh_rest, 3), np.float32)}


def preview(t: dict, tex: np.ndarray, seen: np.ndarray, filled: np.ndarray, path: Path):
    import cv2
    a = np.where(seen[..., None], tex, 0)
    a = np.where((t["inside"] & ~seen)[..., None], np.array([0.35, 0.05, 0.05]), a)
    b = np.where(t["inside"][..., None], filled, 0)
    img = np.concatenate([a, np.ones((a.shape[0], 8, 3)), b], 1)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    f = 1800 / img.shape[1]
    if f < 1 or img.shape[1] < 900:  # shrink big ones, enlarge tiny ones (blocky on purpose)
        img = cv2.resize(img, (int(img.shape[1] * f), max(1, int(img.shape[0] * f))),
                         interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
    # the texture's v axis points up for walls: flip so the picture is the right way up
    if abs(float(t["geo"].v @ np.array([0, 1.0, 0]))) > 0.5 and float(t["geo"].v[1]) > 0:
        img = img[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(ds, scene_dir: Path, view, train_idx: list[int], d: dict, out_dir: Path,
        n_sh_rest: int, device) -> tuple[dict | None, dict]:
    """New Gaussians (numpy arrays, same layout as the splat) for the hidden parts of the
    solid scaffold surfaces, and an info dict."""
    surfaces, why = load_surfaces(scene_dir, ds, list(setting(d, "surface_fill_rules")))
    info: dict = {"surfaces": []}
    if not surfaces:
        info["skipped"] = why
        log.info("Surface fill skipped: %s", why)
        return None, info
    photo_idx = sc.sample_images(train_idx, int(setting(d, "surface_fill_photos")))
    fr0 = ds.frames[train_idx[0]]
    cams = np.array([ds.frames[i]["center"] for i in train_idx])
    texs = []
    for srf in surfaces:
        g = sc.Geo(srf)
        dist = float(np.median(np.linalg.norm(cams - g.c, axis=1)))
        footprint = dist / float(fr0["K"][0, 0])  # one photo pixel on the surface
        t = make_texture(srf, footprint, int(setting(d, "surface_fill_max_texels")))
        if t is not None:
            texs.append(t)
    log.info("Surface fill: projecting %d photos onto %d solid surface(s)", len(photo_idx),
             len(texs))
    observe(texs, ds, photo_idx, view, d, device)
    parts, plans = [], []
    for t in texs:
        H, W = t["H"], t["W"]
        w = t["w"].reshape(H, W)
        seen = (w > 0) & t["inside"]
        tex = np.where(seen[..., None], t["col"].reshape(H, W, 3) / np.maximum(w, 1e-12)[..., None], 0)
        # fill only what is hidden BEHIND something (a bush in front). Where photos see past
        # the plane (an opening, or the plane sits a little in front of the real surface),
        # an opaque fill would cover real content; where no photo looked, there's no evidence.
        occ, beyond = t["occ"].reshape(H, W), t["beyond"].reshape(H, W)
        todo = t["inside"] & ~seen & (occ > 0) & (beyond <= 0.5 * occ)
        srf = t["srf"]
        rec = {"id": srf["id"], "material": srf.get("material", ""), "texels": int(t["inside"].sum()),
               "seen": int(seen.sum()), "hidden": int(todo.sum()),
               "not_filled": int((t["inside"] & ~seen & ~todo).sum()),
               "texel_size": round(t["texel"], 4)}
        if todo.sum() < int(setting(d, "surface_fill_min_hidden")) or seen.sum() < 1000:
            rec["skipped"] = "nothing hidden" if seen.sum() >= 1000 else "too little of it seen"
            info["surfaces"].append(rec)
            continue
        filled, vecs, strength, share = fill_texture(tex, seen, todo)
        rec.update(repeat=[v.tolist() for v in vecs], repeat_strength=round(strength, 3),
                   detail_share=round(share, 3))
        plans.append((t, filled, todo, rec))
        name = "".join(ch if ch.isalnum() else "_" for ch in (srf.get("material") or "surface"))
        preview(t, tex, seen, filled, out_dir / "surface_fill" / f"{srf['id']}_{name[:30]}.jpg")
        info["surfaces"].append(rec)
    hidden_total = sum(int(p[2].sum()) for p in plans)
    if not hidden_total:
        info["skipped"] = "no hidden parts to fill"
        log.info("Surface fill: no hidden parts to fill")
        return None, info
    stride = max(1, int(math.ceil(math.sqrt(hidden_total / int(setting(d, "surface_fill_max_new"))))))
    for t, filled, todo, rec in plans:
        p = bake(t, filled, todo, stride, float(setting(d, "surface_fill_opacity")), n_sh_rest)
        rec["new_gaussians"] = int(len(p["means"]))
        parts.append(p)
        log.info("Surface fill: #%s %s: %d of %d texels hidden, repeat %s (strength %.2f), "
                 "%d new Gaussians", rec["id"], rec["material"] or "", rec["hidden"],
                 rec["texels"], rec["repeat"] or "none", rec["repeat_strength"],
                 rec["new_gaussians"])
    new = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    info["new_gaussians"] = int(len(new["means"]))
    info["stride"] = stride
    (out_dir / "surface_fill").mkdir(parents=True, exist_ok=True)
    (out_dir / "surface_fill" / "surface_fill.json").write_text(json.dumps(info, indent=2))
    return new, info
