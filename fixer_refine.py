#!/usr/bin/env python3
"""
Difix3D+-style refinement of a finished splat with NVIDIA Fixer (AI fill-in: the result
contains generated content, so keep the real-only splat alongside it).

  1. Rebuild the training images from the original video using the scene bundle published
     with every result (<name>/scene/: camera poses, which frames were used, settings).
  2. Load the finished .ply.
  3. Several rounds of:
       - pick camera positions the drone never had: shifted off the flight path and turned
         towards angles between the cube faces, a little further out each round;
       - render the splat there (this is where holes and smears show);
       - NVIDIA Fixer (Cosmos-based single-step diffusion) cleans each render;
       - keep training on the real photos plus these cleaned views (lower weight).
  4. Report held-out PSNR on real photos before/after (each round that makes the splat worse
     on real photos is undone), save preview triplets (render | Fixer | real photo of the same
     surfaces) and the refined .ply.

    python fixer_refine.py --video patio.mp4 --scene patio/scene --ply patio/patio.ply \\
        --out refined/ --work work/

Settings live under `fixer:` in config.yaml (the bundle's config, plus --overrides JSON).
Fixer runs in its own environment via fixer_fix.py. Licence: NVIDIA Open Model License
(commercial use allowed) for the weights, Apache-2.0 for the code.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import appearance  # noqa: E402
import naming  # noqa: E402
import splat_pipeline as sp  # noqa: E402
import train_gsplat as tg  # noqa: E402

log = logging.getLogger("fixer")

FIXER_DEFAULTS = {
    "rounds": 3,               # progressive rounds, each further from the real cameras
    "views_per_round": 200,    # novel views rendered + fixed per round (Fixer: ~20/s on a 5090)
    "steps_per_round": 1500,   # training steps after each round's fixes
    "novel_ratio": 0.3,        # share of training steps that use a fixed novel view
    "novel_weight": 0.3,       # loss weight of novel views relative to real photos
    "max_shift": 0.15,         # final camera offset from the flight path, x scene scale
    "yaw_jitter": 35,          # degrees of extra turn left/right (between cube faces)
    "pitch_jitter": 25,        # degrees of extra tilt up/down
    "render_width": 1024,      # Fixer works at 1024x576 (16:9); novel views are rendered so
    "render_height": 576,
    "hfov": 90,                # horizontal field of view of the novel views, degrees
    "timestep": 250,           # Fixer's single denoising step (NVIDIA's published setting)
    "gate": "raw",             # judge rounds on real photos as viewers see them ("raw"), or
                               # with per-photo colour corrections ("corrected")
    "alpha_mask": 0.0,         # >0: only learn from fixed pixels where the render had this alpha
    "previews": 6,             # preview triplets saved per round
    "ref_candidates": 6,       # photos checked for occlusion when picking preview photos
    "opacity_entropy": 0.003,  # push Gaussians to clearly solid or gone (clears hanging haze)
    "appearance_warmup": "auto",  # steps fitting colour corrections first ("auto": 2 per photo)
    "max_round_drop": 0.3,     # a round that costs more held-out PSNR than this is undone
    "python": "/opt/fixer/bin/python",
    "model": "/opt/fixer/models/pretrained/pretrained_fixer.pkl",
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def read_ply(path: Path) -> dict:
    """Inverse of train_gsplat.write_ply (3DGS binary little-endian float layout)."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            header += line
        names = [l.split()[-1] for l in header.decode().splitlines() if l.startswith("property")]
        n = int(next(l.split()[-1] for l in header.decode().splitlines()
                     if l.startswith("element vertex")))
        data = np.frombuffer(f.read(n * 4 * len(names)), dtype="<f4").reshape(n, len(names))
    col = {nm: i for i, nm in enumerate(names)}
    rest = sorted((nm for nm in names if nm.startswith("f_rest_")), key=lambda s: int(s[7:]))
    k = len(rest) // 3
    get = lambda keys: data[:, [col[x] for x in keys]].astype(np.float32)  # noqa: E731
    shN = get(rest).reshape(n, 3, k).transpose(0, 2, 1) if k else np.zeros((n, 0, 3), np.float32)
    return {"means": get(["x", "y", "z"]),
            "sh0": get(["f_dc_0", "f_dc_1", "f_dc_2"]).reshape(n, 1, 3),
            "shN": np.ascontiguousarray(shN),
            "opacities": get(["opacity"]).reshape(n),
            "scales": get(["scale_0", "scale_1", "scale_2"]),
            "quats": get(["rot_0", "rot_1", "rot_2", "rot_3"])}


def rebuild_images(video: Path, scene: Path, work: Path, cfg: dict) -> Path:
    """Re-cut exactly the training images the bundle's poses belong to."""
    sel = json.loads((scene / "selection.json").read_text())
    cfg = sp.deep_merge(cfg, {"extract": {"fps": sel["extract_fps"], "keep_raw": True}})
    job = sp.Job(video, work, work / "_out", cfg)
    log.info("Re-extracting %d selected frames from %s", len(sel["frames"]), video.name)
    sp.stage_extract(job)
    out = sp.reset_dir(job.frames)
    for frame, raw_name in sel["frames"].items():
        src = job.raw / raw_name
        if not src.exists():
            raise RuntimeError(f"{raw_name} missing after extraction: is this the same video?")
        shutil.copy2(src, out / f"{frame}.jpg")
    shutil.rmtree(job.raw, ignore_errors=True)
    sp.stage_slice(job)
    want = json.loads((scene / "intrinsics.json").read_text())
    got = job.read_json("intrinsics.json")
    if got["views"] != want["views"] or got["width"] != want["width"]:
        raise RuntimeError("Re-cut views don't match the bundle (different slice settings)")
    sp.reset_dir(job.sparse)
    shutil.copytree(scene / "sparse" / "0", job.sparse / "0")
    return job.dir


# --------------------------------------------------------------------------- #
# Novel cameras
# --------------------------------------------------------------------------- #
def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def novel_cameras(ds, train_idx: list[int], n: int, shift: float, up: np.ndarray,
                  d: dict, rng: np.random.Generator) -> list[dict]:
    """Cameras off the flight path. Each starts from a real camera, moves `shift` x scene
    scale in a random direction (mostly sideways, some up/down) and turns by up to
    yaw_jitter/pitch_jitter degrees, so it sees between and beyond the real views."""
    W, H = int(d["render_width"]), int(d["render_height"])
    f = (W / 2) / math.tan(math.radians(float(d["hfov"])) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
    out = []
    for i in rng.choice(train_idx, size=min(n, len(train_idx)), replace=len(train_idx) < n):
        fr = ds.frames[i]
        V = fr["viewmat"].astype(np.float64)
        R_wc = V[:3, :3].T  # camera-to-world rotation
        c = -R_wc @ V[:3, 3]
        v = rng.normal(size=3)
        v -= 0.6 * (v @ up) * up  # favour horizontal moves
        v /= np.linalg.norm(v) + 1e-12
        c_new = c + v * shift * ds.scene_scale
        yaw = rng.uniform(-1, 1) * float(d["yaw_jitter"])
        pitch = rng.uniform(-1, 1) * float(d["pitch_jitter"])
        right = R_wc[:, 0]
        R_new = _rot(up, yaw) @ _rot(right, pitch) @ R_wc
        Vn = np.eye(4)
        Vn[:3, :3] = R_new.T
        Vn[:3, 3] = -R_new.T @ c_new
        out.append({"viewmat": Vn.astype(np.float32), "K": K.copy(), "W": W, "H": H,
                    "center": c_new, "forward": R_new[:, 2], "source": int(i)})
    return out


class ReferencePicker:
    """Chooses, for a new viewpoint, the real photo that shows the same surfaces (shown next
    to the render and Fixer's output in the previews, to judge what Fixer made up).

    The new view's rendered depth gives a few hundred 3D points on what it sees. Every training
    photo is scored by how many of those points fall inside its frame at a similar distance
    (so a photo from far away or facing elsewhere loses), then the best few are rendered from
    their own camera to check the points aren't hidden behind something there (occlusion)."""

    def __init__(self, ds, train_idx: list[int], render_depth, d: dict):
        self.ds, self.idx, self.render_depth = ds, list(train_idx), render_depth
        self.k = max(1, int(d.get("ref_candidates", 6)))
        fr = [ds.frames[i] for i in self.idx]
        self.R = np.stack([f["viewmat"][:3, :3] for f in fr]).astype(np.float64)  # [C,3,3]
        self.t = np.stack([f["viewmat"][:3, 3] for f in fr]).astype(np.float64)   # [C,3]
        self.K = np.stack([f["K"] for f in fr]).astype(np.float64)
        self.WH = np.array([[f["W"], f["H"]] for f in fr], np.float64)

    @staticmethod
    def surface_points(cam: dict, depth: np.ndarray, alpha: np.ndarray, n: int = 24):
        H, W = depth.shape
        vv, uu = np.mgrid[0:H:max(1, H // n), 0:W:max(1, W // n)]
        vv, uu = vv.ravel(), uu.ravel()
        z = depth[vv, uu].astype(np.float64)
        ok = (alpha[vv, uu] > 0.5) & np.isfinite(z) & (z > 0)
        if ok.sum() < 8:
            return None, None
        uu, vv, z = uu[ok] + 0.5, vv[ok] + 0.5, z[ok]
        K, V = cam["K"].astype(np.float64), cam["viewmat"].astype(np.float64)
        xc = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
        return (xc - V[:3, 3]) @ V[:3, :3], z  # world points, their depth in the new view

    def pick(self, cam: dict, depth: np.ndarray, alpha: np.ndarray) -> tuple[int, float]:
        """(photo index, share of the view's surfaces it shows); falls back to the nearest
        photo by position and direction when the render is mostly empty."""
        P, z_new = self.surface_points(cam, depth, alpha)
        if P is None:
            return nearest_photo(self.ds, self.idx, cam), 0.0
        Xc = np.einsum("cij,mj->cmi", self.R, P) + self.t[:, None, :]  # [C,M,3]
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[:, None, 0, 0] * Xc[..., 0] / z + self.K[:, None, 0, 2]
            v = self.K[:, None, 1, 1] * Xc[..., 1] / z + self.K[:, None, 1, 2]
            ratio = z / z_new[None]
        inside = ((z > 1e-6) & (u >= 0) & (v >= 0) & (u < self.WH[:, None, 0])
                  & (v < self.WH[:, None, 1]) & (ratio > 0.5) & (ratio < 2.0))
        cover = inside.mean(1)
        top = np.argsort(-cover)[: self.k]
        best, best_score = int(top[0]), -1.0
        for c in top:
            if cover[c] <= 0:
                break
            sel = inside[c]
            # Visible = the candidate's own rendered depth agrees with the point's distance
            d_img = self.render_depth(self.ds.frames[self.idx[c]])  # low-res depth map
            h, w = d_img.shape
            sx, sy = w / self.WH[c, 0], h / self.WH[c, 1]
            ui = np.clip((u[c, sel] * sx).astype(int), 0, w - 1)
            vi = np.clip((v[c, sel] * sy).astype(int), 0, h - 1)
            zp = z[c, sel]
            visible = np.abs(d_img[vi, ui] - zp) < 0.08 * zp + 1e-6
            score = float(visible.sum()) / len(P)
            if score > best_score:
                best, best_score = int(c), score
        return self.idx[best], best_score


def nearest_photo(ds, train_idx: list[int], cam: dict) -> int:
    """Real training photo closest in position and viewing direction."""
    best, best_score = train_idx[0], 1e18
    for i in train_idx:
        fr = ds.frames[i]
        R_wc = fr["viewmat"][:3, :3].T
        fwd = R_wc[:, 2]
        score = (np.linalg.norm(fr["center"] - cam["center"]) / ds.scene_scale
                 + 2.0 * (1 - float(fwd @ cam["forward"])))
        if score < best_score:
            best, best_score = i, score
    return best


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #
def run_fixer(d: dict, pairs: list[dict], work: Path):
    pairs_file = work / "pairs.json"
    pairs_file.write_text(json.dumps(pairs))
    cmd = [d["python"], str(HERE / "fixer_fix.py"), "--pairs", str(pairs_file),
           "--model", d["model"], "--timestep", str(int(d["timestep"])),
           "--width", str(int(d["render_width"])), "--height", str(int(d["render_height"]))]
    sp.run(cmd, echo=True)


def refine(data_dir: Path, ply: Path, out_dir: Path, s: dict, d: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    device = torch.device("cuda")
    rng = np.random.default_rng(int(s["seed"]))
    random.seed(int(s["seed"]))
    torch.manual_seed(int(s["seed"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_fixer_work"
    work.mkdir(exist_ok=True)

    views = json.loads((data_dir / "intrinsics.json").read_text())["views"]
    ds = tg.ColmapDataset(data_dir, downscale=float(s["downscale"]), align=bool(s["align"]),
                          views=views, up_axis=s["up_axis"], world_scale=float(s["world_scale"]))
    train_idx, test_idx, expo_src = tg.split_train_test(ds, int(s["test_every"]))
    up = (tg.TARGET_AXES[s["up_axis"]][0].astype(np.float64) if ds.alignment is not None
          else np.array([0, -1.0, 0]))  # unaligned: COLMAP's first camera, y down

    g = read_ply(ply)
    n = len(g["means"])
    sh_degree = int(round(math.sqrt(g["shN"].shape[1] + 1))) - 1
    log.info("Loaded %s: %d Gaussians, SH degree %d; %d real training photos, %d held out",
             ply.name, n, sh_degree, len(train_idx), len(test_idx))
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(torch.from_numpy(v)) for k, v in g.items()}).to(device)
    del g

    total = int(d["rounds"]) * int(d["steps_per_round"])
    lrs = {"means": 1.6e-5 * ds.scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": splats[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"],
                                                       gamma=0.1 ** (1.0 / max(total, 1)))
    strategy = MCMCStrategy(cap_max=n, refine_start_iter=200, refine_stop_iter=total)
    strategy.check_sanity(splats, optimizers)
    try:
        state = strategy.initialize_state(scene_scale=ds.scene_scale)
    except TypeError:
        state = strategy.initialize_state()
    use_expo = bool(s["exposure"])
    expo = torch.nn.Parameter(torch.zeros(len(ds), 3, 4, device=device))
    expo_opt = torch.optim.Adam([expo], lr=float(s["exposure_lr"]), eps=1e-15)
    eye3 = torch.eye(3, device=device)
    bil, bil_opt = appearance.make(len(ds), s, device)
    ssim = tg.make_ssim(torch, F, device)
    lam = float(s["ssim_lambda"])
    mode = "antialiased" if s.get("antialiased") else "classic"
    log.info("Rendering mode %s (matches how the splat was trained); bilateral grids %s",
             mode, "on" if bil is not None else "off")

    def render(fr, with_depth=False, scale=1.0):
        K = torch.from_numpy(fr["K"]).to(device).float()
        W, H = fr["W"], fr["H"]
        if scale != 1.0:
            K = K.clone()
            K[:2] *= scale
            W, H = max(8, int(W * scale)), max(8, int(H * scale))
        r, a, info = rasterization(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=torch.from_numpy(fr["viewmat"]).to(device)[None],
            Ks=K[None], width=W, height=H, render_mode="RGB+ED" if with_depth else "RGB",
            sh_degree=sh_degree, near_plane=float(s["near_plane"]), packed=bool(s["packed"]),
            rasterize_mode=mode)
        return r, a, info

    @torch.no_grad()
    def render_depth_small(fr):
        r, a, _ = render(fr, with_depth=True, scale=0.25)
        dep = r[0, ..., 3]  # "ED" is already the expected (alpha-normalised) depth
        return torch.where(a[0, ..., 0] > 0.5, dep, torch.zeros_like(dep)).cpu().numpy()

    def apply_expo(img, i):
        if i is None:
            return img
        if use_expo:
            dd = expo[i]
            img = img @ (eye3 + dd[:, :3]).T + dd[:, 3]
        if bil is not None:
            img = bil(img, i)
        return img

    def appearance_step(i, loss):
        if use_expo and i is not None:
            loss = loss + float(s["exposure_reg"]) * expo[i].pow(2).mean()
        if bil is not None and i is not None:
            loss = loss + float(s["bilagrid_tv"]) * bil.tv_loss(i)
        return loss

    train_idx_t = torch.tensor(train_idx, device=device)

    def appearance_opt_step():
        for o_ in ([expo_opt] if use_expo else []) + ([bil_opt] if bil is not None else []):
            o_.step()
            o_.zero_grad(set_to_none=True)
        # Average correction = no change, so the splat itself keeps the true colours
        if bil is not None:
            bil.recenter_(train_idx_t)
        if use_expo:
            with torch.no_grad():
                expo[train_idx_t] -= expo[train_idx_t].mean(0, keepdim=True)

    @torch.no_grad()
    def evaluate(tag: str):
        """(held-out PSNR with per-photo corrections, without them = what a viewer shows)"""
        if not test_idx:
            return None, None
        ps, raw = [], []
        for i in test_idx:
            img, _ = ds.load(i)
            gt = torch.from_numpy(img).to(device).float().div_(255.0)[None]
            r, _, _ = render(ds.frames[i])
            o = apply_expo(r[..., :3], expo_src.get(i)).clamp(0, 1)
            ps.append(-10 * math.log10(max(F.mse_loss(o, gt).item(), 1e-10)))
            raw.append(-10 * math.log10(max(F.mse_loss(r[..., :3].clamp(0, 1), gt).item(), 1e-10)))
        v, vr = float(np.mean(ps)), float(np.mean(raw))
        log.info("EVAL %s: held-out PSNR %.2f on %d real photos (%.2f without per-photo "
                 "corrections)", tag, v, len(ps), vr)
        return v, vr

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in splats.items()}

    loader = tg.Prefetcher(ds, train_idx, threads=int(s["loader_threads"]), seed=int(s["seed"]))
    # The per-photo colour corrections from the original training aren't stored with the splat,
    # so re-fit them (splat frozen) before scoring: otherwise 'before' would be unfairly low.
    warm = d.get("appearance_warmup", "auto")
    warm = min(4000, 2 * len(train_idx)) if warm == "auto" else int(warm)
    warm = warm if (use_expo or bil is not None) else 0
    for w in range(warm):
        i, img, mask, _ = loader.get()
        gt = torch.from_numpy(img).to(device).float().div_(255.0)[None]
        with torch.no_grad():
            r, _, _ = render(ds.frames[i])
        o = apply_expo(r[..., :3], i)
        if mask is not None:
            m = torch.from_numpy(mask).to(device).float()[None, ..., None]
            l1 = (o - gt).abs().mul(m).sum() / (m.sum() * 3).clamp(min=1)
        else:
            l1 = F.l1_loss(o, gt)
        appearance_step(i, l1).backward()
        appearance_opt_step()
    if warm:
        log.info("Fitted per-photo colour corrections for %d steps (splat frozen)", warm)
    raw_gate = str(d.get("gate", "raw")) == "raw"
    before, before_raw = evaluate("before Fixer")
    stats = {"rounds": [], "before_psnr": before, "before_psnr_raw": before_raw,
             "gate": "raw" if raw_gate else "corrected"}
    best = {"psnr": before_raw if raw_gate else before, "round": 0, "splats": snapshot()}
    picker = ReferencePicker(ds, train_idx, render_depth_small, d) if int(d["previews"]) else None
    ref_scores: list[float] = []
    novel: list[tuple[np.ndarray, dict, np.ndarray]] = []  # (fixed image, camera, render alpha)
    step = 0
    t0 = time.time()
    preview_dir = out_dir / "fixer_preview"
    preview_dir.mkdir(exist_ok=True)
    try:
        for rnd in range(1, int(d["rounds"]) + 1):
            # ---- render novel views and fix them ---------------------------------------
            shift = float(d["max_shift"]) * rnd / int(d["rounds"])
            cams = novel_cameras(ds, train_idx, int(d["views_per_round"]), shift, up, d, rng)
            rdir = work / f"round{rnd}"
            sp.reset_dir(rdir)
            pairs, alphas = [], []
            with torch.no_grad():
                for k, cam in enumerate(cams):
                    want_ref = picker is not None and k < int(d["previews"])
                    r, a, _ = render(cam, with_depth=want_ref)
                    img = (r[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
                    alpha = a[0, ..., 0].float().cpu().numpy()
                    alphas.append(alpha)
                    cv2.imwrite(str(rdir / f"{k:04d}_in.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    pair = {"input": str(rdir / f"{k:04d}_in.png"),
                            "output": str(rdir / f"{k:04d}_fixed.png")}
                    if want_ref:
                        depth = r[0, ..., 3].cpu().numpy()  # expected depth
                        ref_i, shared = picker.pick(cam, depth, alpha)
                        ref_scores.append(shared)
                        pair["ref"] = str(ds.frames[ref_i]["path"])
                    pairs.append(pair)
            torch.cuda.empty_cache()
            log.info("Round %d/%d: %d novel views at %.0f%% of max offset; running Fixer",
                     rnd, int(d["rounds"]), len(cams), 100 * rnd / int(d["rounds"]))
            run_fixer(d, pairs, rdir)
            novel = []
            for k, (cam, pair) in enumerate(zip(cams, pairs)):
                fixed = cv2.imread(pair["output"])
                if fixed is None:
                    continue
                novel.append((cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB), cam, alphas[k]))
                if k < int(d["previews"]):
                    row = [cv2.imread(pair["input"]), fixed]
                    if pair.get("ref"):
                        ref = cv2.imread(pair["ref"])  # square photo: fit, keep its shape
                        h = fixed.shape[0]
                        ref = cv2.resize(ref, (int(ref.shape[1] * h / ref.shape[0]), h))
                        row.append(ref)
                    cv2.imwrite(str(preview_dir / f"round{rnd}_{k:02d}_render_fixer_photo.jpg"),
                                np.concatenate(row, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not novel:
                raise RuntimeError("Fixer produced no images")

            # ---- train on real photos + fixed novel views -------------------------------
            for _ in range(int(d["steps_per_round"])):
                use_novel = random.random() < float(d["novel_ratio"])
                if use_novel:
                    img, fr, alpha = novel[random.randrange(len(novel))]
                    mask, i = None, None
                    if float(d["alpha_mask"]) > 0:
                        mask = alpha > float(d["alpha_mask"])
                else:
                    i, img, mask, _ = loader.get()
                    fr = ds.frames[i]
                gt = torch.from_numpy(img).to(device).float().div_(255.0)[None]
                r, a, info = render(fr)
                o = apply_expo(r[..., :3], i)
                if mask is not None:
                    m = torch.from_numpy(mask).to(device).float()[None, ..., None]
                    l1 = (o - gt).abs().mul(m).sum() / (m.sum() * 3).clamp(min=1)
                    o_l, gt_l = o * m, gt * m
                else:
                    l1 = F.l1_loss(o, gt)
                    o_l, gt_l = o, gt
                loss = (1 - lam) * l1 + lam * (1 - ssim(o_l.permute(0, 3, 1, 2),
                                                         gt_l.permute(0, 3, 1, 2)))
                if use_novel:
                    loss = loss * float(d["novel_weight"])
                opa = torch.sigmoid(splats["opacities"])
                loss = loss + s["opacity_reg"] * opa.mean()
                loss = loss + s["scale_reg"] * torch.exp(splats["scales"]).mean()
                w_ent = float(d.get("opacity_entropy", 0)) * min(1.0, step / 500)
                if w_ent > 0:  # faint, see-through Gaussians become solid or vanish (and MCMC
                    # recycles vanished ones where detail is needed)
                    oc = opa.clamp(1e-4, 1 - 1e-4)
                    loss = loss - w_ent * (oc * oc.log() + (1 - oc) * (1 - oc).log()).mean()
                loss = appearance_step(i, loss)
                strategy.step_pre_backward(params=splats, optimizers=optimizers, state=state,
                                           step=step, info=info)
                loss.backward()
                for o_ in optimizers.values():
                    o_.step()
                    o_.zero_grad(set_to_none=True)
                appearance_opt_step()
                scheduler.step()
                strategy.step_post_backward(params=splats, optimizers=optimizers, state=state,
                                            step=step, info=info, lr=scheduler.get_last_lr()[0])
                step += 1
                if step % 250 == 0:
                    log.info("  step %d/%d  loss %.4f  (%.1f min)", step, total, loss.item(),
                             (time.time() - t0) / 60)
            score_c, score_raw = evaluate(f"after round {rnd}")
            stats["rounds"].append({"round": rnd, "shift": shift, "views": len(novel),
                                    "psnr": score_c, "psnr_raw": score_raw})
            # Judge on what viewers see (no per-photo corrections) unless configured otherwise
            score = score_raw if raw_gate else score_c
            if score is None or best["psnr"] is None or score >= best["psnr"]:
                best = {"psnr": score, "round": rnd, "splats": snapshot()}
            elif score < best["psnr"] - float(d["max_round_drop"]):
                # Views this far from the flight path are hurting the real ones: stop here
                log.warning("Round %d lowered held-out PSNR %.2f -> %.2f; going back to round %d "
                            "and stopping (later rounds go even further from the path)",
                            rnd, best["psnr"], score, best["round"])
                break
    finally:
        loader.close()

    if stats["rounds"] and best["round"] != stats["rounds"][-1]["round"]:
        with torch.no_grad():
            for k in list(splats.keys()):
                splats[k] = torch.nn.Parameter(best["splats"][k].to(device))
    stats["kept_round"] = best["round"]
    kept_r = next((r for r in stats["rounds"] if r["round"] == best["round"]), None)
    stats["after_psnr"] = kept_r["psnr"] if kept_r else before
    stats["after_psnr_raw"] = kept_r["psnr_raw"] if kept_r else before_raw
    del best
    with torch.no_grad():
        kept, dropped = tg.write_ply(
            out_dir / "splat.ply", splats["means"].cpu().numpy(), splats["sh0"].cpu().numpy(),
            splats["shN"].cpu().numpy(), splats["opacities"].cpu().numpy(),
            splats["scales"].cpu().numpy(), splats["quats"].cpu().numpy())
    with torch.no_grad():
        opa = torch.sigmoid(splats["opacities"])
        stats["faint_share"] = round(float(((opa > 0.005) & (opa < 0.3)).float().mean()), 4)
    stats.update({"gaussians": kept, "minutes": round((time.time() - t0) / 60, 1),
                  "settings": d, "generated_content": True,
                  "model": "NVIDIA Fixer (nvidia/Fixer, NVIDIA Open Model License)"})
    (out_dir / "fixer_stats.json").write_text(json.dumps(stats, indent=2, default=float))
    if stats["before_psnr_raw"] is not None and stats["after_psnr_raw"] is not None:
        log.info("Fixer refinement: held-out PSNR on real photos %.2f -> %.2f as viewers see it "
                 "(kept round %d; the gain is mostly in views the drone never saw)",
                 stats["before_psnr_raw"], stats["after_psnr_raw"], stats["kept_round"])
    shutil.rmtree(work, ignore_errors=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True, help="the result's scene/ bundle")
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work_fixer"))
    ap.add_argument("--overrides", default="", help="JSON merged over the bundle's config")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(message)s")
    cfg = yaml.safe_load((args.scene / "config.yaml").read_text()) or {}
    if args.overrides.strip():
        cfg = sp.deep_merge(cfg, json.loads(args.overrides))
    cfg = sp.deep_merge(sp.DEFAULT_CONFIG, cfg)
    if os.environ.get("COLMAP_BIN"):
        cfg["sfm"]["colmap_bin"] = os.environ["COLMAP_BIN"]
    cfg_file = args.work / "fixer_config.yaml"
    args.work.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    s = tg.load_settings(str(cfg_file), {})
    ts = args.scene / "train_settings.json"
    trained = json.loads(ts.read_text()) if ts.exists() else {}
    # Render the way the splat was trained: splats made before anti-aliasing was added are
    # 'classic'; mixing the two makes small details too thick or too faint.
    s["antialiased"] = bool(trained.get("antialiased", False))
    # Per-photo local colour grids let floaters hide behind corrections: never use them here,
    # even when an older bundle's config turned them on.
    s["bilagrid"] = False
    d = {**FIXER_DEFAULTS, **(cfg.get("fixer") or {})}
    data_dir = rebuild_images(args.video, args.scene, args.work, cfg)
    stats = refine(data_dir, args.ply, args.out, s, d)
    print("FIXER_REFINE_DONE " + json.dumps({k: stats[k] for k in (
        "before_psnr_raw", "after_psnr_raw", "kept_round", "gaussians", "minutes")}))


if __name__ == "__main__":
    main()
