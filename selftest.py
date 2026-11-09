"""
Self-test: checks every part of the pipeline in a few minutes, before you commit hours of
GPU time to a real video.

    python selftest.py              # all checks, including a tiny end-to-end run
    python selftest.py --quick      # skip the end-to-end run (about 1 minute)
    python selftest.py --with-fixer # also load and run NVIDIA Fixer (the AI fill-in step)

In Docker, on your own PC (needs the NVIDIA GPU passed through):
    docker run --rm --gpus all <image> python /opt/app/selftest.py

Checks
  environment   user, working folder writable, free disk, model caches present
  tools         ffmpeg, ffprobe and COLMAP run (and whether COLMAP has CUDA)
  gpu           torch sees the GPU; gsplat renders and back-propagates on it
  depth         the depth-prior model loads and runs (only if train.custom.depth_prior)
  mapanything   MapAnything loads (only if sfm.mapper is mapanything)
  fixer         Fixer loads in its own environment and fixes one image (only with --with-fixer)
  mini_run      renders a synthetic 360 video of a textured room and runs the REAL pipeline
                on it (extract, select, slice, COLMAP, train, export) with tiny settings,
                using the same config, binaries and Python as a real run

Exit code 0 = everything passed. Each failure prints the error and what to do about it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import splat_pipeline as sp  # noqa: E402

log = logging.getLogger("selftest")
RESULTS: list[tuple[str, str, float, str]] = []  # (name, PASS/FAIL/SKIP/WARN, seconds, detail)


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def check(name: str):
    def deco(fn):
        def wrapper(*a, **k):
            log.info("---- %s ----", name)
            t0 = time.time()
            try:
                out = fn(*a, **k)
                status, detail = out if isinstance(out, tuple) else ("PASS", out or "")
            except Exception as e:  # report and carry on with the other checks
                status, detail = "FAIL", f"{type(e).__name__}: {e}"
                log.debug(traceback.format_exc())
            dt = time.time() - t0
            RESULTS.append((name, status, dt, str(detail)))
            (log.info if status in ("PASS", "SKIP") else log.error)(
                "%s %s (%.1fs)%s", name, status, dt, f": {detail}" if detail else "")
            return status
        return wrapper
    return deco


def sh(cmd: list, timeout: int = 600, env: dict | None = None) -> str:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                       env=env)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise RuntimeError(f"`{' '.join(map(str, cmd[:3]))} ...` exited {r.returncode}\n{tail}")
    return out


def load_cfg() -> tuple[dict, Path]:
    """Same config a real run uses: config.yaml + COLMAP_BIN + SPLAT_CONFIG_OVERRIDES."""
    base = Path(os.environ.get("SPLAT_CONFIG", APP_DIR / "config.yaml"))
    cfg = yaml.safe_load(base.read_text()) if base.exists() else {}
    if os.environ.get("COLMAP_BIN"):
        cfg = sp.deep_merge(cfg, {"sfm": {"colmap_bin": os.environ["COLMAP_BIN"]}})
    over = os.environ.get("SPLAT_CONFIG_OVERRIDES", "").strip()
    if over:
        cfg = sp.deep_merge(cfg, json.loads(over))
    return sp.deep_merge(sp.DEFAULT_CONFIG, cfg), base


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
@check("environment")
def check_environment(work: Path):
    info = [f"python {sys.version.split()[0]}", f"uid {os.getuid()}" if hasattr(os, "getuid") else ""]
    work.mkdir(parents=True, exist_ok=True)
    probe = work / ".write_test"
    probe.write_bytes(os.urandom(1 << 20))
    probe.unlink()
    free = shutil.disk_usage(work).free / 1e9
    info.append(f"{free:.0f} GB free in {work}")
    warn = []
    if free < 40:
        warn.append(f"only {free:.0f} GB free: a 60 s 8K video can need 30-60 GB")
    for var in ("HF_HOME", "TORCH_HOME"):
        p = os.environ.get(var)
        if p and not Path(p).is_dir():
            warn.append(f"{var}={p} does not exist (baked-in model weights missing?)")
    if os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("HF_HOME"):
        if not any(Path(os.environ["HF_HOME"]).rglob("config.json")):
            warn.append("HF_HUB_OFFLINE=1 but no models found under HF_HOME")
    detail = ", ".join(i for i in info if i)
    return ("WARN", detail + "; " + "; ".join(warn)) if warn else ("PASS", detail)


@check("tools")
def check_tools(cfg: dict):
    found = []
    for tool in ("ffmpeg", "ffprobe"):
        first = sh([tool, "-version"]).splitlines()[0]
        found.append(" ".join(first.split()[:3]))
    colmap = sp.resolve_colmap(cfg["sfm"]["colmap_bin"])
    head = (sh([colmap, "-h"]) or "").strip().splitlines()
    banner = next((l for l in head if "COLMAP" in l), head[0] if head else "?")
    cuda_line = next((l for l in head if "CUDA" in l), "")
    found.append(banner.split("--")[0].strip())
    if cfg["sfm"].get("use_gpu") and "without CUDA" in cuda_line:
        return "WARN", ", ".join(found) + "; COLMAP was built WITHOUT CUDA but sfm.use_gpu is " \
                                          "true: feature extraction will fail or be very slow"
    return "PASS", ", ".join(found) + (f" ({cuda_line.strip()})" if cuda_line else "")


@check("gpu")
def check_gpu():
    out = sh([sys.executable, APP_DIR / "train_gsplat.py", "--check"], timeout=1800)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    gpu = next((l.split("GPU:", 1)[1].strip() for l in lines if "GPU:" in l), "?")
    torch_v = next((l.split("torch ", 1)[1].strip() for l in lines if "torch " in l), "?")
    return "PASS", f"{gpu}; torch {torch_v}; gsplat render + backward OK"


DEPTH_SNIPPET = r"""
import sys, numpy as np, cv2, tempfile, pathlib
sys.path.insert(0, sys.argv[1])
import depth_prior
backend, model = depth_prior.load_model(sys.argv[2], sys.argv[3], int(sys.argv[4]))
img = (np.random.default_rng(0).random((480, 480, 3)) * 255).astype("uint8")
p = pathlib.Path(tempfile.mkdtemp()) / "x.jpg"; cv2.imwrite(str(p), img)
d = model([p])[0]
assert np.isfinite(d).all() and d.size > 0, "depth output not finite"
print("DEPTH_OK", backend, d.shape)
"""


@check("depth")
def check_depth(cfg: dict):
    c = cfg["train"].get("custom") or {}
    if not c.get("depth_prior"):
        return "SKIP", "train.custom.depth_prior is false"
    out = sh([sys.executable, "-c", DEPTH_SNIPPET, APP_DIR, c.get("depth_backend", "auto"),
              c.get("depth_model", ""), c.get("depth_process_res", 756)], timeout=900)
    line = next((l for l in out.splitlines() if l.startswith("DEPTH_OK")), "DEPTH_OK ?")
    return "PASS", "model " + line.split(" ", 2)[1] + " loaded and ran on the GPU"


@check("mapanything")
def check_mapanything(cfg: dict):
    if cfg["sfm"]["mapper"] != "mapanything":
        return "SKIP", f"sfm.mapper is {cfg['sfm']['mapper']}"
    ma = cfg["sfm"]["mapanything"]
    py = ma.get("python") or sys.executable
    name = (ma.get("settings") or {}).get("model", "facebook/map-anything-apache")
    sh([py, "-c", "import sys; from mapanything.models import MapAnything; "
                  f"m = MapAnything.from_pretrained({name!r}); print('MA_OK')"], timeout=900)
    return "PASS", f"{name} loads"


@check("claude")
def check_claude(cfg: dict):
    """The scene-scaffold agent: key present and the model answers (costs a fraction of a cent)."""
    c = cfg["train"].get("custom") or {}
    if not c.get("scaffold", True) or str(c.get("scaffold_mode", "claude")) != "claude":
        return "SKIP", "scaffold_mode isn't claude"
    import scaffold_agent
    s = {"scaffold_model": c.get("scaffold_model", "claude-opus-5-5")}
    client, why = scaffold_agent.make_client(s)
    if client is None:
        return "WARN", f"{why}: the scaffold will use automatic rules"
    r = client.messages.create(model=s["scaffold_model"], max_tokens=5,
                               messages=[{"role": "user", "content": "Reply with just OK"}])
    text = "".join(getattr(b, "text", "") for b in r.content).strip()
    return "PASS", f"{s['scaffold_model']} answered {text[:20]!r}"


@check("fixer")
def check_fixer(enabled: bool):
    if not enabled:
        return "SKIP", "only checked with --with-fixer"
    py = os.environ.get("FIXER_PYTHON", "/opt/fixer/bin/python")
    if not Path(py).exists():
        raise RuntimeError(f"{py} not found: the image was built without Fixer (WITH_FIXER=0)")
    out = sh([py, APP_DIR / "fixer_fix.py", "--probe"], timeout=1800)
    line = next((l for l in out.splitlines() if l.startswith("FIXER_OK")), "")
    if not line:
        raise RuntimeError("fixer_fix.py --probe did not report FIXER_OK\n" + out[-2000:])
    note = next((l for l in out.splitlines() if "PyTorch stand-in" in l or "retrying" in l), "")
    return "PASS", "Fixer loaded and ran: " + line.split(" ", 1)[1] + (
        " (slower PyTorch path: " + note[:120] + ")" if note else "")


# --------------------------------------------------------------------------- #
# Synthetic 360 video: a textured box room seen from a moving camera
# --------------------------------------------------------------------------- #
def _face_texture(rng, size=1024):
    import cv2
    img = (rng.random((size, size, 3)) * 60 + 100).astype(np.uint8)
    for _ in range(900):  # blobs and boxes of random colour and size: plenty of SIFT corners
        c = tuple(int(x) for x in rng.integers(0, 256, 3))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        r = int(rng.integers(4, 40))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(rng.integers(4, 40))), c, -1)
    noise = (rng.random((size, size, 3)) * 40 - 20)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def render_room_video(path: Path, frames: int = 11, width: int = 2048, fps: int = 6):
    """Equirectangular frames (same convention as splat_pipeline's slicer: x right, y down,
    z forward) of a 6 x 3 x 5 m room, camera sliding 1.6 m sideways and 0.6 m forward."""
    import cv2
    rng = np.random.default_rng(7)
    H, W = width // 2, width
    tex = [_face_texture(rng) for _ in range(6)]
    half = np.array([3.0, 1.5, 2.5])  # room half-extents (x, y, z)
    ppm = 160.0  # texture pixels per metre
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)
    d = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    d = d.reshape(-1, 3)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(frames):
        s = k / max(1, frames - 1)
        o = np.array([-0.8 + 1.6 * s, 0.2 * math.sin(2 * math.pi * s), -0.3 + 0.6 * s])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_axes = np.where(d > 0, (half - o) / d, (-half - o) / d)
        t_axes[~np.isfinite(t_axes)] = np.inf
        axis = np.argmin(t_axes, 1)
        t = t_axes[np.arange(len(d)), axis]
        hit = o + d * t[:, None]
        sign = d[np.arange(len(d)), axis] > 0
        face = axis * 2 + sign
        out = np.zeros((len(d), 3), np.uint8)
        other = {0: (2, 1), 1: (0, 2), 2: (0, 1)}  # in-plane axes per wall normal
        for f in range(6):
            sel = face == f
            if not sel.any():
                continue
            a, b = other[f // 2]
            u = ((hit[sel, a] + half[a]) * ppm) % tex[f].shape[1]
            v = ((hit[sel, b] + half[b]) * ppm) % tex[f].shape[0]
            out[sel] = tex[f][v.astype(int), u.astype(int)]
        proc.stdin.write(out.reshape(H, W, 3).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg could not encode the synthetic video (libx264 missing?)")


@check("mini_run")
def check_mini_run(cfg: dict, work: Path, until: str | None, cpu_colmap: bool):
    root = work / "_selftest"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    video = root / "selftest_room.mp4"
    render_room_video(video)
    tiny = {
        "extract": {"fps": 6, "max_frames": 0, "keep_raw": False},
        "select": {"target_fps": 6, "min_sharpness": 0},
        "slice": {"size": 400, "workers": 4},
        "sfm": {"max_features": 4096, "matcher": "sequential"},
        "train": {"iterations": 300,
                  "custom": {"cap_max": 100000, "eval_every": 150, "save_every": 1000,
                             "test_every": 10, "depth_batch": 4}},
    }
    if cpu_colmap:
        tiny["sfm"]["use_gpu"] = False
    run_cfg = sp.deep_merge(cfg, tiny)
    cfg_file = root / "selftest_config.yaml"
    cfg_file.write_text(yaml.safe_dump({k: v for k, v in run_cfg.items()
                                        if not k.startswith("_")}, sort_keys=False))
    run_cfg = sp.load_config(cfg_file)
    res = sp.run_pipeline([video], run_cfg, root / "work", root / "out",
                          until_stage=until, skip_gate=True)["selftest_room.mp4"]
    job_dir = Path(res["work_dir"])
    stages = {}
    for m in sorted((job_dir / ".stages").glob("*.done")):
        stages[m.stem] = json.loads(m.read_text()).get("seconds")
    if res["status"] != "ok":
        log_tail = ""
        plog = job_dir / "pipeline.log"
        if plog.exists():
            log_tail = "\n".join(plog.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"pipeline {res['status']} after stages {list(stages)}:\n"
                           f"{res['error']}\n--- pipeline.log tail ---\n{log_tail}")
    parts = [f"stages {stages}"]
    st = job_dir / "sfm_stats.json"
    if st.exists():
        s = json.loads(st.read_text())
        parts.append(f"SfM registered {s.get('registered_ratio', 0):.0%} via "
                     f"{s.get('mapper_used', s.get('mapper', cfg['sfm']['mapper']))}")
    status = "PASS"
    if until is None:
        ply = root / "out" / "selftest_room.ply"
        header = ply.read_bytes()[:400].decode("ascii", "replace")
        n = next((int(l.split()[-1]) for l in header.splitlines()
                  if l.startswith("element vertex")), 0)
        if n == 0:
            raise RuntimeError(f"{ply} has no Gaussians")
        parts.append(f"splat with {n} Gaussians")
        ts = job_dir / "train" / "train_stats.json"
        if ts.exists():
            dp = json.loads(ts.read_text()).get("depth_prior")
            if isinstance(dp, dict) and dp.get("error"):
                status = "WARN"
                parts.append(f"trained WITHOUT depth prior ({dp['error']})")
    return status, "; ".join(parts)


# --------------------------------------------------------------------------- #
HINTS = {
    "claude": "The Anthropic API call failed: check ANTHROPIC_API_KEY (console.anthropic.com), "
              "your credit, and train.custom.scaffold_model. Runs still work (automatic scaffold).",
    "environment": "Working folder not writable or low on disk. Set SPLAT_WORK to a bigger folder.",
    "tools": "ffmpeg/COLMAP missing or broken in the image: rebuild with the full Dockerfile.",
    "gpu": "No usable GPU. Locally: docker run --gpus all ...; in Foundry: check the GPU resource "
           "queue. If torch/gsplat fail to import, the PyTorch stack in the image is broken "
           "(Dockerfile.quick repairs it).",
    "depth": "The depth model can't load. Training still runs without it; to silence this set "
             "train.custom.depth_prior: false, or rebuild with Dockerfile.quick.",
    "mapanything": "MapAnything can't load; the pipeline will fall back to COLMAP.",
    "fixer": "Fixer can't load or run. It needs a bfloat16 GPU (RTX 30xx or newer, A100...) "
             "and an image built with WITH_FIXER=1 (the default); the main "
             "pipeline doesn't need it.",
    "mini_run": "A stage failed on synthetic data: the log tail above shows which and why.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the end-to-end mini run")
    ap.add_argument("--work", type=Path, help="scratch folder (default: SPLAT_WORK or a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the mini run's files")
    ap.add_argument("--until", choices=sp.STAGES, help="stop the mini run after this stage")
    ap.add_argument("--with-fixer", action="store_true", help="also check NVIDIA Fixer")
    ap.add_argument("--only-fixer", action="store_true", help="check only Fixer")
    ap.add_argument("--cpu-colmap", action="store_true", help="run COLMAP on the CPU in the mini run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(message)s")
    work = args.work or Path(os.environ.get("SPLAT_WORK") or tempfile.mkdtemp(prefix="splat_st_"))
    cfg, cfg_path = load_cfg()
    log.info("Self-test using config %s, work folder %s", cfg_path, work)
    t0 = time.time()

    if args.only_fixer:
        check_environment(work)
        check_fixer(True)
        args.quick = True
    else:
        check_environment(work)
        check_tools(cfg)
        gpu_ok = check_gpu() == "PASS"
        check_depth(cfg)
        check_mapanything(cfg)
        check_claude(cfg)
        check_fixer(args.with_fixer)
    if args.quick:
        RESULTS.append(("mini_run", "SKIP", 0.0, "--quick"))
    elif not gpu_ok and not args.until:
        RESULTS.append(("mini_run", "SKIP", 0.0, "GPU check failed, so training can't run"))
    else:
        check_mini_run(cfg, work, args.until, args.cpu_colmap)
    if not args.keep:
        shutil.rmtree(work / "_selftest", ignore_errors=True)

    print("\n================ SELF-TEST SUMMARY ================", flush=True)
    for name, status, dt, detail in RESULTS:
        lines = [l for l in detail.splitlines() if l.strip()]
        short = lines[0] if lines else ""
        if status == "FAIL" and len(lines) > 1:  # the command line, then the actual error
            errs = [l for l in lines[1:] if "Error" in l or "error" in l]
            short = (errs[-1] if errs else lines[-1]).strip()
        print(f"{status:5} {name:12} {dt:6.1f}s  {short[:160]}")
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    for name, _, _, _ in failed:
        print(f"  -> {name}: {HINTS.get(name, '')}")
    print(f"{'ALL CHECKS PASSED' if not failed else f'{len(failed)} CHECK(S) FAILED'} "
          f"in {time.time() - t0:.0f}s", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
