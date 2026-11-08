"""
Runs NVIDIA Difix (single-step diffusion that removes artefacts from rendered views of a 3D
reconstruction) on a batch of images. Runs in its own Python environment (/opt/difix) because
Difix pins old diffusers/transformers versions; torch is shared with the main environment.

    python difix_fix.py --pairs pairs.json      # [{"input": ..., "ref": ..., "output": ...}, ...]
    python difix_fix.py --probe                 # load the model and fix one image (self-test)

LICENCE: Difix (code and weights) is under the NVIDIA License: non-commercial research and
evaluation use only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIFIX_SRC = Path(os.environ.get("DIFIX_SRC", "/opt/difix/src"))
sys.path.insert(0, str(DIFIX_SRC))

# The model ships its own code, which diffusers copies into HF_MODULES_CACHE (default
# $HF_HOME/modules) before importing it. The baked-in /opt/hf is read-only at run time.
if "HF_MODULES_CACHE" not in os.environ:
    _hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if not os.access(_hf, os.W_OK):
        import tempfile
        os.environ["HF_MODULES_CACHE"] = tempfile.mkdtemp(prefix="hf_modules_")

PROMPT = "remove degradation"


def load(model: str, dtype_name: str):
    import torch
    from pipeline_difix import DifixPipeline  # from the Difix3D repo (src/pipeline_difix.py)
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    pipe = DifixPipeline.from_pretrained(model, trust_remote_code=True, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def fix_one(pipe, img, ref, size: int, timestep: int):
    import numpy as np
    from PIL import Image
    w0, h0 = img.size
    kw = dict(prompt=PROMPT, image=img.resize((size, size), Image.BICUBIC),
              num_inference_steps=1, timesteps=[timestep], guidance_scale=0.0,
              height=size, width=size)
    if ref is not None:
        kw["ref_image"] = ref.resize((size, size), Image.BICUBIC)
    out = pipe(**kw).images[0]
    arr = np.asarray(out, dtype=np.float32)
    if not np.isfinite(arr).all() or arr.max() < 1:  # fp16 overflow shows up as black/NaN
        raise FloatingPointError("Difix produced an empty image")
    return out.resize((w0, h0), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default=os.environ.get("DIFIX_MODEL", "nvidia/difix_ref"))
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--timestep", type=int, default=199)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    t0 = time.time()
    dtype = args.dtype
    pipe = load(args.model, dtype)
    print(f"DIFIX loaded {args.model} ({dtype}) in {time.time() - t0:.0f}s on "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    if args.probe:
        rng = np.random.default_rng(0)
        img = Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8"))
        try:
            fix_one(pipe, img, img, 256, args.timestep)
        except FloatingPointError:
            del pipe
            torch.cuda.empty_cache()
            pipe, dtype = load(args.model, "fp32"), "fp32"
            fix_one(pipe, img, img, 256, args.timestep)
        print(f"DIFIX_OK dtype={dtype} peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.1f}",
              flush=True)
        return

    pairs = json.loads(args.pairs.read_text())
    t1 = time.time()
    for k, p in enumerate(pairs):
        img = Image.open(p["input"]).convert("RGB")
        ref = Image.open(p["ref"]).convert("RGB") if p.get("ref") else None
        try:
            out = fix_one(pipe, img, ref, args.size, args.timestep)
        except FloatingPointError:
            if dtype == "fp32":
                raise
            print("DIFIX fp16 produced an empty image; switching to fp32", flush=True)
            del pipe
            torch.cuda.empty_cache()
            pipe, dtype = load(args.model, "fp32"), "fp32"
            out = fix_one(pipe, img, ref, args.size, args.timestep)
        out.save(p["output"])
        if (k + 1) % 25 == 0 or k + 1 == len(pairs):
            rate = (k + 1) / (time.time() - t1)
            print(f"DIFIX {k + 1}/{len(pairs)} images ({rate:.2f}/s)", flush=True)
    print(json.dumps({"difix_done": len(pairs), "dtype": dtype,
                      "seconds": round(time.time() - t0, 1),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}),
          flush=True)


if __name__ == "__main__":
    main()
