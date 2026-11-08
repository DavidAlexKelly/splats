"""
Shared conventions for the pinhole views sliced out of each 360 frame.

Image names
    v03/f00012.jpg   view 3 of 360 frame f00012. One folder per view ("sensor") and the same
                     file name across folders for the same frame: the layout COLMAP's rig
                     support expects (see https://colmap.github.io/rigs.html).
    f00012_v03.jpg   the older flat layout; still understood so existing work folders resume.

View layouts (slice.layout in config.yaml)
    cubemap   6 non-overlapping 90 deg faces: front/right/back/left/up/down. Least redundant
              data for training; slice.cubemap_faces can drop faces (e.g. up for open sky).
    custom    the explicit slice.views list of [yaw, pitch] pairs (overlapping views).
"""
from __future__ import annotations

import math
import re

CUBEMAP_FACES = {  # name -> [yaw, pitch] in degrees (positive pitch looks up)
    "front": [0, 0],
    "right": [90, 0],
    "back": [180, 0],
    "left": [270, 0],
    "up": [0, 90],
    "down": [0, -90],
}

_FOLDER = re.compile(r"(?:^|/)v(\d+)/([^/]+?)\.[A-Za-z]+$")
_FLAT = re.compile(r"(?:^|/)([^/]+?)_v(\d+)\.[A-Za-z]+$")
_DIGITS = re.compile(r"(\d+)")


def resolve_views(slice_cfg: dict) -> list[list[float]]:
    """The [yaw, pitch] list for a slice config (layout cubemap or custom)."""
    layout = (slice_cfg.get("layout") or "custom").lower()
    if layout == "cubemap":
        faces = slice_cfg.get("cubemap_faces") or list(CUBEMAP_FACES)
        unknown = [f for f in faces if f not in CUBEMAP_FACES]
        if unknown:
            raise ValueError(f"Unknown cubemap face(s) {unknown}; use {list(CUBEMAP_FACES)}")
        return [list(CUBEMAP_FACES[f]) for f in faces]
    if layout != "custom":
        raise ValueError(f"Unknown slice.layout '{layout}' (use cubemap or custom)")
    views = slice_cfg.get("views") or []
    if not views:
        raise ValueError("slice.layout is custom but slice.views is empty")
    return [list(v[:2]) for v in views]


def image_name(frame_stem: str, view_idx: int) -> str:
    return f"v{view_idx:02d}/{frame_stem}.jpg"


def view_folder(view_idx: int) -> str:
    return f"v{view_idx:02d}"


def parse(name: str) -> tuple[str, int] | None:
    """'v03/f00012.jpg' or 'f00012_v03.jpg' -> ('f00012', 3); None if neither."""
    m = _FOLDER.search(name)
    if m:
        return m.group(2), int(m.group(1))
    m = _FLAT.search(name)
    if m:
        return m.group(1), int(m.group(2))
    return None


def frame_number(frame_stem: str) -> int:
    m = _DIGITS.findall(frame_stem)
    return int(m[-1]) if m else 0


def flat_key(name: str) -> str:
    """A unique, folder-free file stem for an image (cache and preview file names)."""
    stem = name.rsplit(".", 1)[0]
    return stem.replace("/", "__").replace("\\", "__")


def rig_from_cam(yaw_deg: float, pitch_deg: float):
    """Rotation taking camera-frame rays (x right, y down, z forward) to the level 360 frame:
    pitch about x first, then yaw about y. Matches splat_pipeline.build_view_maps()."""
    import numpy as np
    p, t = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    return Ry @ Rx


def views_overlap(views: list, fov_deg: float) -> bool:
    """True if any two views share pixels: their axes are closer than one FOV apart
    (a cube map at exactly 90 deg only shares edges, so it counts as non-overlapping)."""
    import numpy as np
    dirs = [rig_from_cam(*v[:2]) @ np.array([0, 0, 1.0]) for v in views]
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(dirs[i] @ dirs[j])))))
            if ang < fov_deg - 1e-3:
                return True
    return False


def rotmat_to_qvec(R) -> list[float]:
    """3x3 rotation -> [w, x, y, z] (Shepperd's method, w >= 0)."""
    import numpy as np
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return [float(x) for x in q]


def colmap_rig_config(views: list, ref: int = 0) -> list[dict]:
    """COLMAP rig_config.json for the sliced views: one rig, one sensor per view folder,
    all sharing the 360 camera's centre with the known relative rotations."""
    import numpy as np
    pano_from_cam = [rig_from_cam(*v[:2]) for v in views]
    cams = []
    for i, R in enumerate(pano_from_cam):
        entry: dict = {"image_prefix": view_folder(i) + "/"}
        if i == ref:
            entry["ref_sensor"] = True
        else:
            # cam_i <- pano <- cam_ref
            cam_from_rig = R.T @ pano_from_cam[ref]
            entry["cam_from_rig_rotation"] = rotmat_to_qvec(cam_from_rig)
            entry["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cams.append(entry)
    return [{"cameras": cams}]
