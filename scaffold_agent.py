"""
Claude builds the scene scaffold (stage 2 of the scaffold idea).

The pipeline does the measuring (scaffold.py: candidate surfaces, photo checks, outlines);
Claude does the judging, through tools:

  show_candidates   candidates highlighted in their best photos
  show_photo        any photo, optionally zoomed, with candidates highlighted
  photo_index       contact sheet of photos along the flight
  edit_region       remove / add / split off part of an outline, as seen in a photo
  merge             join candidates that are the same surface
  add_surface       a surface the pipeline missed, from a region in a photo (glass: fitted to
                    its frame, because depth sees through glass)
  set_rules         solid / glass / soft / ignore, with material, confidence, reason
  review            the whole scaffold drawn over photos and from above (self-check)
  finish            done (asks for one review first if anything changed since the last one)

build() is the entry point the trainer calls. It reuses a saved decision when one matches
this reconstruction, otherwise finds candidates, applies the automatic rules, and (when an
Anthropic API key is available) lets Claude decide. Any failure falls back to the automatic
rules: a run never fails because of the agent.

Costs are capped (scaffold_max_cost, estimated from token counts) and the decision is saved
(scaffold_decision.json) so re-runs don't pay again. Edit that file by hand to override.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

import scaffold as sc

log = logging.getLogger("train")
APP_DIR = Path(__file__).resolve().parent
MAX_SIDE = 768  # photos sent to Claude


# --------------------------------------------------------------------------- #
# Small image helpers
# --------------------------------------------------------------------------- #
def _jpeg_b64(img_rgb: np.ndarray, quality: int = 85) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _img_block(img_rgb: np.ndarray) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": _jpeg_b64(img_rgb)}}


def _fit(img: np.ndarray, side: int = MAX_SIDE) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    f = side / max(h, w)
    if f >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)


def _grid(img: np.ndarray) -> np.ndarray:
    """Faint 0.1 grid with edge labels, so Claude can give coordinates."""
    out = img.copy()
    h, w = out.shape[:2]
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        out[:, x] = (out[:, x].astype(np.int32) * 0.6 + 255 * 0.4).astype(np.uint8)
        out[y, :] = (out[y, :].astype(np.int32) * 0.6 + 255 * 0.4).astype(np.uint8)
        sc._label(out, f".{k}", x + 2, 12, 0.35)
        sc._label(out, f".{k}", 2, y - 2, 0.35)
    return out


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    bar = np.full((24, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text[:90], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def _sheet(tiles: list[np.ndarray], cols: int, tile: int) -> np.ndarray:
    import cv2
    rows = math.ceil(len(tiles) / cols)
    th = tile + 24
    out = np.full((rows * th, cols * tile, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        f = tile / max(t.shape[:2])
        t2 = cv2.resize(t, (int(t.shape[1] * f), int(t.shape[0] * f)), interpolation=cv2.INTER_AREA) \
            if f < 1 else t
        r, c = divmod(k, cols)
        out[r * th:r * th + t2.shape[0], c * tile:c * tile + t2.shape[1]] = t2[:th]
    return out


def _region_mask(W: int, H: int, box=None, polygon=None) -> np.ndarray:
    import cv2
    m = np.zeros((H, W), np.uint8)
    if polygon:
        pts = np.array([[float(x) * W, float(y) * H] for x, y in polygon], np.int32)
        cv2.fillPoly(m, [pts], 1)
    elif box:
        x0, y0, x1, y1 = (float(v) for v in box)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        m[int(y0 * H):max(int(y1 * H), int(y0 * H) + 1),
          int(x0 * W):max(int(x1 * W), int(x0 * W) + 1)] = 1
    return m.astype(bool)


# --------------------------------------------------------------------------- #
# The workspace Claude operates on
# --------------------------------------------------------------------------- #
class Workspace:
    def __init__(self, ds, surfaces: list[dict], s: dict, load_prior, image_idx: list[int],
                 up: np.ndarray, out_dir: Path):
        self.ds, self.s, self.load_prior = ds, s, load_prior
        self.surfaces = surfaces
        self.image_idx = image_idx  # photos used for checks
        self.up = up
        self.out_dir = out_dir
        self.cams = np.array([f["center"] for f in ds.frames])
        self.name_to_idx = {f["name"]: k for k, f in enumerate(ds.frames)}
        self.next_id = max([x["id"] for x in surfaces], default=-1) + 1
        self.changed_since_review = True
        self.reviews = 0
        self.log_lines: list[str] = []
        typ = surfaces[0]["tol"] / float(s["scaffold_tol"]) if surfaces else 1.0
        self.tol = float(s["scaffold_tol"]) * typ
        self.cell = float(s["scaffold_cell"]) * typ

    # ---- lookups -----------------------------------------------------------
    def by_id(self, sid: int) -> dict:
        for x in self.surfaces:
            if x["id"] == int(sid):
                return x
        raise ValueError(f"no candidate #{sid}")

    def frame(self, name: str) -> int:
        name = str(name).strip()
        if name in self.name_to_idx:
            return self.name_to_idx[name]
        for k, f in enumerate(self.ds.frames):  # tolerate missing extension / folder
            if f["name"].endswith(name) or Path(f["name"]).stem == Path(name).stem:
                return k
        raise ValueError(f"no photo named {name!r} (use photo_index to list them)")

    def photo(self, i: int) -> np.ndarray:
        img, _ = self.ds.load(i)
        return img

    def prior(self, i: int):
        p = self.load_prior(i)
        return None if p is None else p[0]

    # ---- descriptions ------------------------------------------------------
    def describe(self, srf: dict) -> str:
        st = srf["stats"]
        parts = [f"#{srf['id']} {srf['label']}", f"area {st.get('area', 0):.2f}",
                 f"points {st.get('points', 0)}", f"conc {st.get('concentration', 0):.2f}",
                 f"photo_agree {st.get('photo_agree', 0):.2f}",
                 f"see_through {st.get('see_through', 0):.2f}",
                 f"hidden {st.get('hidden', 0):.2f}", f"views {st.get('views', 0)}",
                 f"auto={srf.get('auto_rule')}"]
        if srf.get("decided_by") == "claude":
            parts.append(f"RULE={srf['rule']}")
        return "  ".join(parts)

    def table(self) -> str:
        rows = sorted(self.surfaces, key=lambda x: -x["stats"].get("area", 0))
        return "\n".join(self.describe(x) for x in rows)

    # ---- pictures ----------------------------------------------------------
    def view(self, i: int, ids=None, by_rule=False, box=None, grid=True) -> np.ndarray:
        img = self.photo(i)
        shown = self.surfaces if ids != [] else []
        out = sc.draw_on_photo(img, self.ds.frames[i], shown, by_rule, prior=self.prior(i),
                               ids_only=ids if ids else None,
                               show_hidden=bool(ids)) if shown else img
        if grid:
            out = _grid(out)
        if box:
            H, W = out.shape[:2]
            x0, y0, x1, y1 = (float(v) for v in box)
            x0, x1 = sorted((max(0.0, x0), min(1.0, x1)))
            y0, y1 = sorted((max(0.0, y0), min(1.0, y1)))
            out = out[int(y0 * H):max(int(y1 * H), int(y0 * H) + 8),
                      int(x0 * W):max(int(x1 * W), int(x0 * W) + 8)]
            import cv2
            f = MAX_SIDE / max(out.shape[:2])
            if f > 1:
                out = cv2.resize(out, (int(out.shape[1] * f), int(out.shape[0] * f)),
                                 interpolation=cv2.INTER_CUBIC)
        return _fit(out)

    def topdown(self, by_rule: bool, title: str = "") -> np.ndarray:
        import cv2
        img = sc.draw_topdown(self.surfaces, self.ds.points, self.cams, self.up, by_rule,
                              size=1000, title=title)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def gallery(self, ids: list[int], tile: int = 320) -> list[np.ndarray]:
        tiles = []
        for sid in ids:
            srf = self.by_id(sid)
            if not srf.get("best_views"):
                continue
            i = srf["best_views"][0]
            v = sc.draw_on_photo(_fit(self.photo(i), tile * 2), self.ds.frames[i], self.surfaces,
                                 False, prior=self.prior(i), ids_only=[sid], alpha=0.5,
                                 show_hidden=True)
            tiles.append(_caption(_fit(v, tile), f"#{sid} {srf['label']} in "
                                                 f"{self.ds.frames[i]['name']}"))
        return [_sheet(tiles[k:k + 16], 4, tile) for k in range(0, len(tiles), 16)]

    def overview_photos(self, n: int = 12) -> np.ndarray:
        """Horizontal views spread along the flight, labelled with their names."""
        import naming
        horiz = [k for k, f in enumerate(self.ds.frames)
                 if (naming.parse(f["name"]) or ("", 0))[1] in (0, 1, 2, 3)] \
            or list(range(len(self.ds.frames)))
        pick = sc.sample_images(horiz, n)
        tiles = [_caption(_fit(self.photo(i), 384), self.ds.frames[i]["name"]) for i in pick]
        return _sheet(tiles, 4, 384)

    # ---- edits -------------------------------------------------------------
    def _region_hits(self, g: sc.Geo, i: int, box, polygon, grow: bool):
        """In-plane coords of the plane seen through the region in photo i."""
        fr = self.ds.frames[i]
        W, H = 512, int(round(512 * fr["H"] / fr["W"]))
        m = _region_mask(W, H, box, polygon)
        if not m.any():
            raise ValueError("empty region: give box [x0, y0, x1, y1] or polygon [[x, y], ...] "
                             "as fractions of the photo")
        o, dirs, uu, vv = sc.pixel_rays(fr, 4, W=W, H=H)
        sel = m[np.clip(vv, 0, H - 1), np.clip(uu, 0, W - 1)]
        z, _, uv = sc.hit(g, o, dirs, grid_only=False)
        ok = sel & np.isfinite(z) & (z > 0)
        # beyond 30x the typical distance the plane is nearly edge-on: ignore
        ok &= z < 30 * self.tol / float(self.s["scaffold_tol"])
        return uv[ok], z[ok], (uu[ok], vv[ok], W, H)

    def edit_region(self, sid: int, image: str, action: str, box=None, polygon=None) -> str:
        srf = self.by_id(sid)
        g = sc.Geo(srf)
        i = self.frame(image)
        if action == "add":
            uv, _, _ = self._region_hits(g, i, box, polygon, True)
            if len(uv) == 0:
                return f"#{sid}'s plane isn't visible in that region of {image}; nothing added"
            g.grow_to(uv)
            r, c, ok = g.cell_of(uv)
            before = int(g.mask.sum())
            g.mask[r[ok], c[ok]] = True
            g.mask = sc.clean_mask(g.mask, close=1)
            g.save()
            self._restat(srf)
            self.changed_since_review = True
            return f"#{sid}: added {int(g.mask.sum()) - before} cells. Now: {self.describe(srf)}"
        # remove / split: outline cells that project into the region (and aren't hidden)
        cc = g.cell_centres()
        if len(cc) == 0:
            return f"#{sid} has an empty outline"
        fr = self.ds.frames[i]
        W, H = 512, int(round(512 * fr["H"] / fr["W"]))
        x, y, z = sc.project_points(fr, cc, W, H)
        m = _region_mask(W, H, box, polygon)
        inside = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xi, yi = np.clip(x, 0, W - 1).astype(int), np.clip(y, 0, H - 1).astype(int)
        inside &= m[yi, xi]
        pri = self.prior(i)
        if pri is not None:
            import cv2
            d = cv2.resize(pri, (W, H), interpolation=cv2.INTER_NEAREST)[yi, xi]
            visible = (d <= 0) | (z <= d * 1.10) | (srf.get("rule") == "glass")
            inside &= visible
        if not inside.any():
            return f"no visible part of #{sid} inside that region of {image}; nothing changed"
        rr, cc_ = np.nonzero(g.mask)
        cut = np.zeros_like(g.mask)
        cut[rr[inside], cc_[inside]] = True
        g.mask &= ~cut
        g.save()
        msg = f"#{sid}: removed {int(cut.sum())} cells"
        if action == "split":
            new = copy.deepcopy(srf)
            new["id"] = self.next_id
            self.next_id += 1
            g2 = sc.Geo(new)
            g2.mask = cut
            g2.save()
            for k in ("rule", "material", "confidence", "reason", "decided_by"):
                new[k] = None if k in ("rule", "confidence") else ""
            new["point_idx"] = []
            self.surfaces.append(new)
            self._restat(new)
            msg += f" into new candidate #{new['id']}: {self.describe(new)}"
        self._restat(srf)
        self.changed_since_review = True
        return msg + f"\nNow: {self.describe(srf)}"

    def merge(self, ids: list[int]) -> str:
        if len(ids) < 2:
            raise ValueError("merge needs at least two ids")
        srcs = [self.by_id(k) for k in ids]
        pts = []
        for x in srcs:
            g = sc.Geo(x)
            cc = g.cell_centres()
            pts.append(cc)
            if x.get("point_idx"):
                pts.append(self.ds.points[np.asarray(x["point_idx"], np.int64)].astype(np.float64))
        P = np.concatenate(pts)
        n, c = sc._fit_plane(P)
        n0 = np.asarray(srcs[0]["normal"])
        if n @ n0 < 0:
            n = -n
        angles = [math.degrees(math.acos(min(1.0, abs(float(n @ np.asarray(x["normal"]))))))
                  for x in srcs]
        resid = float(np.median(np.abs((P - c) @ n)))
        if max(angles) > 20:
            return (f"not merged: the candidates face different ways (up to {max(angles):.0f} "
                    "degrees apart from a common plane)")
        cells = np.concatenate([sc.Geo(x).cell_centres() for x in srcs])
        u, v = sc._basis(n, self.up)
        uv = np.stack([(cells - c) @ u, (cells - c) @ v], 1)
        label = sc.orientation(n, self.up, self.cams, c, float(self.s["scaffold_max_tilt_deg"]))
        new = sc.new_surface(self.next_id, n, c, self.up, self.tol, self.cell, uv, None, label)
        self.next_id += 1
        g = sc.Geo(new)
        r, cc, ok = g.cell_of(uv)
        g.mask[r[ok], cc[ok]] = True
        g.mask = sc.clean_mask(g.mask, close=1)
        g.save()
        pidx = sorted({int(p) for x in srcs for p in x.get("point_idx", [])})
        new["point_idx"] = pidx
        new["stats"] = {"points": len(pidx),
                        "concentration": max(x["stats"].get("concentration", 0) for x in srcs),
                        "merged_from": [x["id"] for x in srcs]}
        self.surfaces[:] = [x for x in self.surfaces if x["id"] not in {s_["id"] for s_ in srcs}]
        self.surfaces.append(new)
        self._restat(new)
        self.changed_since_review = True
        warn = (f" (warning: the merged surface is not very flat: median offset "
                f"{resid / self.tol:.1f}x the plane thickness)") if resid > 1.5 * self.tol else ""
        return f"merged {ids} into #{new['id']}{warn}: {self.describe(new)}"

    def add_surface(self, image: str, fit_from: str = "inside", box=None, polygon=None,
                    rule: str | None = None, material: str = "", reason: str = "") -> str:
        import cv2
        i = self.frame(image)
        fr = self.ds.frames[i]
        W, H = 512, int(round(512 * fr["H"] / fr["W"]))
        m = _region_mask(W, H, box, polygon)
        if not m.any():
            raise ValueError("empty region")
        pts = self.ds.points.astype(np.float64)
        x, y, z = sc.project_points(fr, pts, W, H)
        ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xi, yi = np.clip(x, 0, W - 1).astype(int), np.clip(y, 0, H - 1).astype(int)
        pri = self.prior(i)
        d = cv2.resize(pri, (W, H), interpolation=cv2.INTER_NEAREST) if pri is not None else None
        if d is not None:
            dd = d[yi, xi]
            ok &= (dd <= 0) | (z <= dd * 1.08)  # not hidden behind something
        if fit_from == "frame":
            span = max(int(math.sqrt(m.sum()) * 0.15), 3)
            k = np.ones((2 * span + 1, 2 * span + 1), np.uint8)
            outer = cv2.dilate(m.astype(np.uint8), k).astype(bool)
            inner = cv2.erode(m.astype(np.uint8), k).astype(bool)
            region = outer & ~inner
        else:
            region = m
        sel = ok & region[yi, xi]
        P = pts[sel]
        src = f"{len(P)} 3D points {'around the edge of' if fit_from == 'frame' else 'inside'} the region"
        if len(P) < 15 and fit_from != "frame" and d is not None:
            o, dirs, uu, vv = sc.pixel_rays(fr, 4, W=W, H=H)
            inside_px = m[np.clip(vv, 0, H - 1), np.clip(uu, 0, W - 1)]
            dz = d[np.clip(vv, 0, H - 1), np.clip(uu, 0, W - 1)]
            good = inside_px & (dz > 0)
            P = o + dirs[good] * dz[good][:, None]
            src = f"{len(P)} depth-estimate samples inside the region (few 3D points there)"
        if len(P) < 15:
            return (f"not enough 3D information to place a surface there ({src}). For glass use "
                    "fit_from 'frame' with a region whose edges follow the frame; or pick a "
                    "photo where the surface is closer/clearer.")
        rng = np.random.default_rng(len(P))
        best, best_n = None, 0
        for _ in range(300):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            inl = np.abs((P - a) @ n) < 1.5 * self.tol
            if inl.sum() > best_n:
                best, best_n = inl, int(inl.sum())
        if best is None or best_n < 10:
            return f"couldn't fit a flat surface to {src}"
        n, c = sc._fit_plane(P[best])
        if n @ (fr["center"] - c) < 0:
            n = -n  # normal faces the camera
        g_probe = sc.Geo(sc.new_surface(-1, n, c, self.up, self.tol, self.cell,
                                        np.zeros((1, 2)), None, ""))
        uv, _, _ = self._region_hits(g_probe, i, box, polygon, True)
        if len(uv) == 0:
            return "the fitted plane is edge-on to this photo; try another photo"
        label = sc.orientation(n, self.up, self.cams, c, float(self.s["scaffold_max_tilt_deg"]))
        new = sc.new_surface(self.next_id, n, c, self.up, self.tol, self.cell, uv, None, label)
        self.next_id += 1
        g = sc.Geo(new)
        r, cc, okc = g.cell_of(uv)
        g.mask[r[okc], cc[okc]] = True
        g.mask = sc.clean_mask(g.mask, close=1)
        g.save()
        new["stats"] = {"points": int(best_n), "concentration": round(best_n / len(P), 3),
                        "added_from": image}
        self.surfaces.append(new)
        self._restat(new)
        if rule:
            self.set_rules([{"id": new["id"], "rule": rule, "material": material,
                             "reason": reason or "added by Claude", "confidence": 0.7}])
        self.changed_since_review = True
        return (f"added #{new['id']} ({label}) fitted to {src}; {best_n} fit the plane. "
                f"{self.describe(new)}")

    def set_rules(self, decisions: list[dict]) -> str:
        done, errs = [], []
        for d in decisions:
            try:
                srf = self.by_id(int(d["id"]))
            except (KeyError, ValueError, TypeError) as e:
                errs.append(str(e))
                continue
            r = str(d.get("rule", "")).lower()
            if r not in sc.RULES:
                errs.append(f"#{d.get('id')}: unknown rule {r!r}")
                continue
            srf.update(rule=r, material=str(d.get("material", ""))[:60],
                       confidence=float(d.get("confidence", 0.7)),
                       reason=str(d.get("reason", ""))[:300], decided_by="claude")
            done.append(f"#{srf['id']}={r}")
        self.changed_since_review = True
        return ("set " + ", ".join(done) if done else "nothing set") + \
            (("; errors: " + "; ".join(errs)) if errs else "")

    def _restat(self, srf: dict):
        """Refresh a surface's photo stats without regrowing its outline."""
        st = srf.setdefault("stats", {})
        keep = {k: st[k] for k in ("points", "concentration", "merged_from", "added_from")
                if k in st}
        tmp = copy.deepcopy(srf)
        sc.photo_check([tmp], self.ds, self.load_prior, self.image_idx, self.s, grow=False)
        srf["stats"] = {**tmp["stats"], **keep}
        srf["best_views"] = tmp.get("best_views", [])

    # ---- review ------------------------------------------------------------
    def review_images(self) -> list[np.ndarray]:
        live = [x for x in self.surfaces if x.get("rule") not in (None, "ignore")]
        chosen: list[int] = []
        covered: set = set()
        for x in sorted(live, key=lambda t: -t["stats"].get("area", 0)):
            if x["id"] in covered:
                continue
            for i in x.get("best_views", [])[:1]:
                if i not in chosen:
                    chosen.append(i)
            covered.add(x["id"])
            if len(chosen) >= 8:
                break
        if len(chosen) < 8:  # fill with spread-out horizontal views
            for i in sc.sample_images(self.image_idx, 8):
                if i not in chosen and len(chosen) < 8:
                    chosen.append(i)
        tiles = [_caption(self.view(i, by_rule=True, grid=False), self.ds.frames[i]["name"])
                 for i in chosen]
        return [self.topdown(True, "scaffold"), _sheet(tiles, 4, 384)]

    def review_text(self) -> str:
        by = {r: [] for r in sc.RULES}
        for x in self.surfaces:
            by.setdefault(x.get("rule") or "ignore", []).append(
                f"#{x['id']}{(' ' + x['material']) if x.get('material') else ''}")
        return "\n".join(f"{r}: {', '.join(v) or '-'}" for r, v in by.items())


# --------------------------------------------------------------------------- #
# Tools (Anthropic tool definitions)
# --------------------------------------------------------------------------- #
REGION = {
    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
            "description": "[x0, y0, x1, y1] as fractions of the full photo"},
    "polygon": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                           "minItems": 2, "maxItems": 2},
                "description": "[[x, y], ...] fractions of the full photo (instead of box)"},
}
TOOLS = [
    {"name": "show_candidates",
     "description": "Show candidates highlighted in their best photos (each in its own view).",
     "input_schema": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "integer"}},
         "views_each": {"type": "integer", "description": "photos per candidate (1-3)"}},
         "required": ["ids"]}},
    {"name": "show_photo",
     "description": "Show one photo (with a 0.1 grid), optionally zoomed into a box, with "
                    "candidates highlighted: 'all' (default), 'none', 'rules' (current "
                    "rules colours), or a list of ids.",
     "input_schema": {"type": "object", "properties": {
         "image": {"type": "string"},
         "box": REGION["box"],
         "highlight": {"description": "'all', 'none', 'rules' or a list of ids"}},
         "required": ["image"]}},
    {"name": "photo_index",
     "description": "Contact sheet of photos spread along the flight with their names. "
                    "view: 'horizontal' (v00-v03), 'down' (v05), 'up' (v04) or 'all'.",
     "input_schema": {"type": "object", "properties": {
         "view": {"type": "string"}, "page": {"type": "integer"}}}},
    {"name": "edit_region",
     "description": "Change a candidate's outline using a region of a photo: 'remove' the "
                    "visible part inside the region, 'add' the plane seen through the region, "
                    "or 'split' the visible part inside the region off as a new candidate.",
     "input_schema": {"type": "object", "properties": {
         "id": {"type": "integer"}, "image": {"type": "string"},
         "action": {"type": "string", "enum": ["remove", "add", "split"]}, **REGION},
         "required": ["id", "image", "action"]}},
    {"name": "merge",
     "description": "Merge candidates that are the same surface into one (refits the plane).",
     "input_schema": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["ids"]}},
    {"name": "add_surface",
     "description": "Add a surface the pipeline missed, from a region in one photo. "
                    "fit_from 'frame' fits the plane to 3D points around the region's edge "
                    "(use for glass: depth sees through it); 'inside' fits to points inside.",
     "input_schema": {"type": "object", "properties": {
         "image": {"type": "string"},
         "fit_from": {"type": "string", "enum": ["inside", "frame"]}, **REGION,
         "rule": {"type": "string", "enum": list(sc.RULES)},
         "material": {"type": "string"}, "reason": {"type": "string"}},
         "required": ["image", "fit_from"]}},
    {"name": "set_rules",
     "description": "Set rules for one or more candidates.",
     "input_schema": {"type": "object", "properties": {
         "decisions": {"type": "array", "items": {"type": "object", "properties": {
             "id": {"type": "integer"},
             "rule": {"type": "string", "enum": list(sc.RULES)},
             "material": {"type": "string"},
             "confidence": {"type": "number"},
             "reason": {"type": "string"}}, "required": ["id", "rule"]}}},
         "required": ["decisions"]}},
    {"name": "review",
     "description": "See the whole scaffold (current rules) from above and over photos.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "finish",
     "description": "Done. Give a one-paragraph summary of the scene and your choices.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"]}},
]


def run_tool(ws: Workspace, name: str, args: dict) -> tuple[list[dict], bool]:
    """Returns (tool_result content blocks, finished)."""
    def text(t):
        return {"type": "text", "text": t}

    if name == "show_candidates":
        blocks = []
        for sid in list(args.get("ids") or [])[:6]:
            srf = ws.by_id(sid)
            n = max(1, min(3, int(args.get("views_each") or 2)))
            vs = srf.get("best_views", [])[:n]
            blocks.append(text(ws.describe(srf) + (" (not confirmed by any photo)" if not vs else "")))
            for i in vs:
                blocks.append(text(f"#{sid} in {ws.ds.frames[i]['name']}"))
                blocks.append(_img_block(ws.view(i, ids=[sid])))
        return blocks, False
    if name == "show_photo":
        i = ws.frame(args["image"])
        hl = args.get("highlight", "all")
        if hl == "none":
            img = ws.view(i, ids=[], box=args.get("box"))
        elif hl == "rules":
            img = ws.view(i, by_rule=True, box=args.get("box"))
        elif isinstance(hl, list):
            img = ws.view(i, ids=[int(k) for k in hl], box=args.get("box"))
        else:
            img = ws.view(i, box=args.get("box"))
        note = f"{ws.ds.frames[i]['name']}"
        if args.get("box"):
            note += (f" zoomed to {args['box']} (coordinates you give tools are still fractions "
                     "of the FULL photo)")
        return [text(note), _img_block(img)], False
    if name == "photo_index":
        import naming
        want = {"horizontal": (0, 1, 2, 3), "down": (5,), "up": (4,)}.get(
            str(args.get("view", "horizontal")), None)
        idx = [k for k, f in enumerate(ws.ds.frames)
               if want is None or (naming.parse(f["name"]) or ("", 0))[1] in want] \
            or list(range(len(ws.ds.frames)))
        page = max(0, int(args.get("page") or 0))
        per = 24
        pick = sc.sample_images(idx, per * 4)[page * per:(page + 1) * per]
        if not pick:
            return [text("no more photos")], False
        tiles = [_caption(_fit(ws.photo(i), 256), ws.ds.frames[i]["name"]) for i in pick]
        return [text(f"page {page} ({len(pick)} photos)"), _img_block(_sheet(tiles, 6, 256))], False
    if name == "edit_region":
        return [text(ws.edit_region(int(args["id"]), args["image"], args["action"],
                                    args.get("box"), args.get("polygon")))], False
    if name == "merge":
        return [text(ws.merge([int(k) for k in args["ids"]]))], False
    if name == "add_surface":
        return [text(ws.add_surface(args["image"], args.get("fit_from", "inside"),
                                    args.get("box"), args.get("polygon"), args.get("rule"),
                                    args.get("material", ""), args.get("reason", "")))], False
    if name == "set_rules":
        return [text(ws.set_rules(list(args.get("decisions") or [])))], False
    if name == "review":
        ws.changed_since_review = False
        ws.reviews += 1
        imgs = ws.review_images()
        return [text("Current scaffold (colours = rules):\n" + ws.review_text()),
                *[_img_block(_fit(im, 1200)) for im in imgs]], False
    if name == "finish":
        if ws.changed_since_review:
            ws.changed_since_review = False
            ws.reviews += 1
            imgs = ws.review_images()
            return [text("Before finishing, check your scaffold below (colours = rules). Fix "
                         "anything wrong, then call finish again.\n" + ws.review_text()),
                    *[_img_block(_fit(im, 1200)) for im in imgs]], False
        ws.summary = str(args.get("summary", ""))
        return [text("ok")], True
    raise ValueError(f"unknown tool {name}")


# --------------------------------------------------------------------------- #
# The conversation
# --------------------------------------------------------------------------- #
def _block_to_dict(b) -> dict:
    if isinstance(b, dict):
        return b
    if hasattr(b, "model_dump"):
        d = b.model_dump(exclude_none=True)
        if d.get("type") == "text" and not d.get("citations"):
            d.pop("citations", None)
        return d
    return dict(b.__dict__)


def _set_cache_point(msgs: list[dict]):
    """Prompt caching without ever editing earlier turns (the API checks that Claude's
    earlier reasoning still sits in the exact conversation it was written in): one cache
    point on the briefing, one rolling point on the newest message."""
    for m in msgs[1:]:
        if isinstance(m["content"], list):
            for c in m["content"]:
                if isinstance(c, dict):
                    c.pop("cache_control", None)
    last = msgs[-1]
    if len(msgs) > 1 and last["role"] == "user" and isinstance(last["content"], list) \
            and last["content"]:
        last["content"][-1]["cache_control"] = {"type": "ephemeral"}


def _drop_thinking(msgs: list[dict]) -> int:
    """Remove Claude's reasoning blocks from earlier turns (recovery path)."""
    n = 0
    for m in msgs:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            keep = [c for c in m["content"]
                    if c.get("type") not in ("thinking", "redacted_thinking")]
            n += len(m["content"]) - len(keep)
            m["content"] = keep or [{"type": "text", "text": "(continuing)"}]
    return n


def load_prompt(s: dict) -> str:
    for p in (s.get("scaffold_prompt") or "", os.environ.get("SPLAT_SCAFFOLD_PROMPT", ""),
              "/workspace/scaffold_prompt.md", str(APP_DIR / "scaffold_prompt.md")):
        if p and Path(p).is_file():
            return Path(p).read_text()
    raise FileNotFoundError("scaffold_prompt.md not found")


def make_client(s: dict):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    kf = Path(os.environ.get("SPLAT_ANTHROPIC_KEY_FILE", "/workspace/.anthropic_key"))
    if not key and kf.is_file():  # a key saved on the pod's volume works too
        key = kf.read_text().strip()
    if not key:
        return None, "no ANTHROPIC_API_KEY set (and no /workspace/.anthropic_key)"
    try:
        import anthropic
    except ImportError:
        return None, "the anthropic package isn't installed"
    return anthropic.Anthropic(api_key=key, max_retries=4, timeout=300), "ok"


class Usage:
    def __init__(self, s: dict):
        self.pin = float(s["scaffold_price_in"]) / 1e6
        self.pout = float(s["scaffold_price_out"]) / 1e6
        self.tin = self.tout = self.cw = self.cr = 0

    def add(self, u):
        if u is None:
            return
        g = (lambda k: int(getattr(u, k, 0) or (u.get(k, 0) if isinstance(u, dict) else 0) or 0))
        self.tin += g("input_tokens")
        self.tout += g("output_tokens")
        self.cw += g("cache_creation_input_tokens")
        self.cr += g("cache_read_input_tokens")

    @property
    def cost(self) -> float:
        return (self.tin * self.pin + self.cw * self.pin * 1.25 + self.cr * self.pin * 0.1
                + self.tout * self.pout)

    def as_dict(self):
        return {"input_tokens": self.tin, "output_tokens": self.tout,
                "cache_write_tokens": self.cw, "cache_read_tokens": self.cr,
                "estimated_cost_usd": round(self.cost, 3)}


def converse(ws: Workspace, client, s: dict, scene_name: str) -> dict:
    """Run the agent loop. Returns info; the decisions are in ws.surfaces."""
    system = load_prompt(s)
    model = str(s["scaffold_model"])
    max_turns = int(s["scaffold_max_turns"])
    cap = float(s["scaffold_max_cost"])
    thinking = int(s["scaffold_thinking"])
    usage = Usage(s)
    order = [x["id"] for x in sorted(ws.surfaces, key=lambda x: -x["stats"].get("area", 0))]
    import naming
    n_pos = len({(naming.parse(f["name"]) or (f["name"], 0))[0] for f in ws.ds.frames})
    intro = (f"Scene: '{scene_name}'. {len(ws.ds.frames)} photos from {n_pos} camera "
             f"positions; {len(ws.ds.points)} 3D points. {len(ws.surfaces)} candidate surfaces "
             "(largest first):\n" + ws.table())
    content = [{"type": "text", "text": intro},
               {"type": "text", "text": "Top-down map of the candidates:"},
               _img_block(_fit(ws.topdown(False, "candidates"), 1000)),
               {"type": "text", "text": "Photos along the flight:"},
               _img_block(ws.overview_photos())]
    for k, sheet in enumerate(ws.gallery(order)):
        content += [{"type": "text", "text": f"Candidate gallery {k + 1} (each candidate in its "
                                             "best photo):"}, _img_block(sheet)]
    content[-1]["cache_control"] = {"type": "ephemeral"}
    msgs: list[dict] = [{"role": "user", "content": content}]
    transcript = [f"# Scaffold agent: {scene_name}\n", "## Briefing\n", intro, ""]
    finished, nudges, turns, stop = False, 0, 0, ""
    use_thinking = thinking > 0
    recovered = False
    while turns < max_turns:
        if usage.cost >= cap:
            stop = f"cost cap reached (~${usage.cost:.2f} of ${cap:.2f})"
            break
        turns += 1
        _set_cache_point(msgs)
        kw = dict(model=model, max_tokens=(thinking + 6000) if use_thinking else 6000,
                  system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  tools=TOOLS, messages=msgs)
        if use_thinking:
            kw["thinking"] = {"type": "enabled", "budget_tokens": thinking}
        try:
            resp = client.messages.create(**kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if use_thinking and "thinking" in msg and "signature" not in msg:
                log.info("Scaffold agent: the model doesn't take this thinking setting; "
                         "continuing without it")
                use_thinking = False
                turns -= 1
                continue
            if ("signature" in msg or "thinking" in msg) and not recovered:
                n = _drop_thinking(msgs)
                log.info("Scaffold agent: the API rejected earlier reasoning blocks; dropped %d "
                         "and retrying", n)
                recovered = True
                turns -= 1
                continue
            raise
        usage.add(getattr(resp, "usage", None))
        blocks = [_block_to_dict(b) for b in resp.content]
        msgs.append({"role": "assistant", "content": blocks})
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        for b in blocks:
            if b.get("type") == "text" and b.get("text", "").strip():
                transcript.append(f"**Claude:** {b['text'].strip()}\n")
        if not calls:
            if nudges >= 2:
                stop = "Claude stopped without calling finish"
                break
            nudges += 1
            msgs.append({"role": "user", "content": [{"type": "text", "text":
                         "Please continue with the tools, and call finish when done."}]})
            continue
        results = []
        for b in calls:
            args = b.get("input") or {}
            transcript.append(f"- `{b['name']}` {json.dumps(args)[:400]}")
            try:
                out, done = run_tool(ws, b["name"], args)
            except Exception as e:  # noqa: BLE001  (tell Claude, let it adjust)
                out, done = [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}], False
            for o in out:
                if o.get("type") == "text":
                    transcript.append(f"  - {o['text'][:600]}")
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out})
            finished = finished or done
        msgs.append({"role": "user", "content": results})
        log.info("Scaffold agent: turn %d, %s, ~$%.2f so far", turns,
                 ", ".join(b["name"] for b in calls), usage.cost)
        if finished:
            break
    if not finished and not stop:
        stop = f"turn limit ({max_turns}) reached"
    info = {"model": model, "turns": turns, "finished": finished, "stopped": stop or None,
            "reviews": ws.reviews, "summary": getattr(ws, "summary", ""), **usage.as_dict()}
    transcript += ["", "## Outcome", json.dumps(info, indent=1)]
    (ws.out_dir / "scaffold_agent_log.md").write_text("\n".join(transcript))
    return info


# --------------------------------------------------------------------------- #
# Entry point used by the trainer and by `splat --scaffold-only`
# --------------------------------------------------------------------------- #
def build(ds, s: dict, out_dir: Path, load_prior, train_idx: list[int], up: np.ndarray,
          scene_name: str = "scene", client=None) -> tuple[list[dict], dict]:
    """Decide the scaffold. Returns (surfaces with rules, info). Never raises for agent
    problems: falls back to the automatic rules."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = np.array([f["center"] for f in ds.frames])
    fp = sc.fingerprint(ds.points, ds.alignment)
    info: dict = {"fingerprint": fp}
    dec_path = str(s.get("scaffold_decision") or "").strip()
    own = out_dir / "scaffold_decision.json"
    if not dec_path and own.exists() and not s.get("scaffold_redo"):
        dec_path = str(own)  # an interrupted run of this scene: don't pay for Claude twice
    if dec_path:
        dec_path = str(Path(dec_path).resolve())
        surfaces, why = sc.load_decision(Path(dec_path), fp)
        if surfaces is not None:
            log.info("Scaffold: reusing the decision in %s (%d surfaces)", dec_path, len(surfaces))
            info.update(source="saved decision", decision_file=dec_path)
            meta = json.loads(Path(dec_path).read_text())
            info["agent"] = meta.get("agent")
            import shutil
            for extra in ("scaffold_review.jpg", "scaffold_agent_log.md", "scaffold_candidates.jpg"):
                src = Path(dec_path).parent / extra
                if src.exists() and src.resolve() != (out_dir / extra).resolve():
                    shutil.copy2(src, out_dir / extra)  # keep Claude's review with the result
            if Path(dec_path).resolve() != own.resolve():
                sc.save_decision(own, surfaces, {"fingerprint": fp, "agent": meta.get("agent"),
                                                 "source": "reused " + dec_path,
                                                 "scene": meta.get("scene")})
            sc.save_report(surfaces, ds.points, cams, up, out_dir, {"source": info["source"]})
            info["rules"] = {r: sum(1 for x in surfaces if x.get("rule") == r) for r in sc.RULES}
            return surfaces, info
        log.warning("Scaffold: not using %s: %s", dec_path, why)
        info["decision_rejected"] = why

    surfaces, cinfo = sc.find_candidates(ds.points, up, cams, s, seed=int(s["seed"]))
    info["candidates"] = cinfo
    check_idx = images_with_prior(train_idx, load_prior, int(s["scaffold_check_images"]))
    sc.photo_check(surfaces, ds, load_prior, check_idx, s)
    sc.resolve_overlaps(surfaces, s)
    sc.auto_rules(surfaces, s)
    try:
        cv2.imwrite(str(out_dir / "scaffold_candidates.jpg"),
                    sc.draw_topdown(surfaces, ds.points, cams, up, by_rule=False,
                                    title="candidates"))
    except Exception as e:  # noqa: BLE001
        log.warning("Couldn't draw the candidates: %s", e)
    info["source"] = "automatic"
    mode = str(s["scaffold_mode"]).lower()
    if mode == "claude" and surfaces:
        why = "ok"
        if client is None:
            client, why = make_client(s)
        if client is None:
            log.info("Scaffold: Claude not used (%s); automatic rules instead", why)
            info["agent"] = {"used": False, "why": why}
        else:
            ws = Workspace(ds, surfaces, s, load_prior, check_idx, up, out_dir)
            snapshot = copy.deepcopy(surfaces)
            t0 = time.time()
            try:
                log.info("Scaffold: asking Claude (%s) to review %d candidate(s)",
                         s["scaffold_model"], len(surfaces))
                a = converse(ws, client, s, scene_name)
                surfaces = ws.surfaces
                sc.auto_rules(surfaces, s, only_unset=True)  # undecided: keep auto
                a["seconds"] = round(time.time() - t0, 1)
                info["agent"] = {"used": True, **a}
                info["source"] = "claude"
                try:
                    imgs = [_pad_to(im, 1536) for im in ws.review_images()]
                    cv2.imwrite(str(out_dir / "scaffold_review.jpg"),
                                cv2.cvtColor(np.concatenate(imgs, 0), cv2.COLOR_RGB2BGR))
                except Exception as e:  # noqa: BLE001
                    log.warning("Couldn't draw the review sheet: %s", e)
                log.info("Scaffold: Claude finished in %d turn(s), %.0f s, ~$%.2f%s", a["turns"],
                         a["seconds"], a["estimated_cost_usd"],
                         f" ({a['stopped']})" if a.get("stopped") else "")
            except Exception as e:  # noqa: BLE001  (never lose a run over the agent)
                log.warning("Scaffold: Claude failed (%s: %s); automatic rules instead",
                            type(e).__name__, e)
                surfaces = snapshot
                info["agent"] = {"used": False, "why": f"{type(e).__name__}: {e}"}
    meta = {"fingerprint": fp, "agent": info.get("agent"), "source": info["source"],
            "scene": scene_name}
    sc.save_decision(out_dir / "scaffold_decision.json", surfaces, meta)
    sc.save_report(surfaces, ds.points, cams, up, out_dir, {"source": info["source"],
                                                            "agent": info.get("agent")})
    n_rule = {r: sum(1 for x in surfaces if x.get("rule") == r) for r in sc.RULES}
    log.info("Scaffold (%s): %s", info["source"],
             ", ".join(f"{v} {k}" for k, v in n_rule.items() if v) or "nothing")
    info["rules"] = n_rule
    return surfaces, info


def images_with_prior(train_idx: list[int], load_prior, n: int) -> list[int]:
    """n photos spread along the flight that have a depth estimate (a photo without one
    can't confirm or reject anything)."""
    pool = sc.sample_images(train_idx, n * 4)
    have = [i for i in pool if load_prior(i) is not None]
    if len(have) < n:  # few priors: look through everything
        seen = set(pool)
        have += [i for i in train_idx if i not in seen and load_prior(i) is not None]
        have.sort()
    return sc.sample_images(have, n)


def _pad_to(img: np.ndarray, width: int) -> np.ndarray:
    """Scale an image to at most `width` wide and pad it with white to exactly `width`."""
    import cv2
    if img.shape[1] > width:
        f = width / img.shape[1]
        img = cv2.resize(img, (width, max(1, int(img.shape[0] * f))), interpolation=cv2.INTER_AREA)
    if img.shape[1] == width:
        return img
    pad = np.full((img.shape[0], width - img.shape[1], 3), 255, np.uint8)
    return np.concatenate([img, pad], 1)
