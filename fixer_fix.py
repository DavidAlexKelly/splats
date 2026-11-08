"""
Runs NVIDIA Fixer (a Cosmos-Predict2 0.6B single-step diffusion model that removes artefacts
from rendered views of a 3D reconstruction) on a batch of images. Runs in its own Python
environment (/opt/fixer) because Fixer pins its own diffusers/transformers versions; torch is
shared with the main environment.

    python fixer_fix.py --pairs pairs.json     # [{"input": ..., "output": ...}, ...]
    python fixer_fix.py --probe                # load the model and fix one image (self-test)

Fixer code: github.com/nv-tlabs/Fixer (Apache-2.0), copied to /opt/fixer/src at build time.
Weights: huggingface.co/nvidia/Fixer (NVIDIA Open Model License: commercial use allowed),
in /opt/fixer/models. Needs a GPU with bfloat16 (Ampere or newer: RTX 30xx/40xx/50xx, A100...).

Compared with NVIDIA's own container this environment leaves out two things Fixer never uses
for single-image inference, replacing them with small stand-ins:
  * megatron-core (multi-GPU bookkeeping): single-process answers (1 GPU, rank 0);
  * the Qwen/T5 text encoders (Fixer runs unconditioned, use_text_encoder=False).
If NVIDIA Transformer Engine is missing or broken, a pure-PyTorch stand-in for the three
pieces the model uses (RMSNorm, rotary embedding, attention) takes over: same maths, a little
slower. FIXER_ATTENTION=torch forces that path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

FIXER_SRC = Path(os.environ.get("FIXER_SRC", "/opt/fixer/src"))
FIXER_MODELS = Path(os.environ.get("FIXER_MODELS", "/opt/fixer/models"))
PIX2PIX = "pix2pix_turbo_nocond_cosmos_base_faster_tokenizer"


# --------------------------------------------------------------------------- stand-ins
def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def _stub_text_encoders():
    """Fixer never encodes text, but cosmos-predict2 imports its Qwen VLM text encoder module,
    which insists on flash-attn at import time."""
    class _Unused:
        def __init__(self, *a, **k):
            raise RuntimeError("text encoders aren't installed (Fixer doesn't use them)")
    _module("imaginaire.networks.qwen2_5_vl", Qwen2_5_VisionTransformerPretrainedModel=_Unused,
            Qwen2_5_VLModel=_Unused, get_rope_index=_Unused)
    _module("imaginaire.models.vlm_qwen", build_tokenizer=_Unused, QwenModel=_Unused)
    _module("imaginaire.models.vlm_qwen_omni", QwenVLBaseModel=_Unused)


def _stub_megatron():
    """cosmos-predict2 only asks megatron-core about multi-GPU layout: one process, one GPU."""
    if "megatron.core.parallel_state" in sys.modules:
        return
    one = lambda *a, **k: 1  # noqa: E731
    zero = lambda *a, **k: 0  # noqa: E731
    none = lambda *a, **k: None  # noqa: E731

    def _no_parallel(*a, **k):
        raise RuntimeError("multi-GPU parallelism isn't available in this environment")

    ps = _module(
        "megatron.core.parallel_state",
        is_initialized=lambda: False, model_parallel_is_initialized=lambda: False,
        get_data_parallel_world_size=one, get_context_parallel_world_size=one,
        get_tensor_model_parallel_world_size=one, get_pipeline_model_parallel_world_size=one,
        get_data_parallel_rank=zero, get_context_parallel_rank=zero,
        get_tensor_model_parallel_rank=zero, get_pipeline_model_parallel_rank=zero,
        get_context_parallel_group=none, get_data_parallel_group=none,
        initialize_model_parallel=_no_parallel, sequence_parallel=False)

    class ModelParallelConfig:
        def __init__(self, *a, **k):
            pass
    core = _module("megatron.core", parallel_state=ps, ModelParallelConfig=ModelParallelConfig)
    core.__path__ = []
    meg = _module("megatron", core=core)
    meg.__path__ = []


def _torch_te_standin():
    """Pure-PyTorch versions of the Transformer Engine pieces the Cosmos DiT uses, with TE's
    maths (RMSNorm computed in float32; TE's non-interleaved rotary embedding)."""
    import torch

    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, zero_centered_gamma=False, **_):
            super().__init__()
            self.eps, self.zero_centered = eps, zero_centered_gamma
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))

        def reset_parameters(self):
            with torch.no_grad():
                self.weight.fill_(0.0 if self.zero_centered else 1.0)

        def forward(self, x):
            xf = x.float()
            y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            w = self.weight.float() + (1.0 if self.zero_centered else 0.0)
            return (y * w).to(x.dtype)

    def _rotate_half(x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(t, freqs, tensor_format="sbhd", interleaved=False, fused=False,
                             cu_seqlens=None, cp_size=1, cp_rank=0, start_positions=None):
        if interleaved or tensor_format not in ("sbhd", "bshd") or start_positions is not None:
            raise NotImplementedError("rotary embedding variant not supported by the stand-in")
        cur = t.shape[1] if tensor_format == "bshd" else t.shape[0]
        f = freqs[:cur]
        if tensor_format == "bshd":
            f = f.transpose(0, 1)
        cos_, sin_ = torch.cos(f).to(t.dtype), torch.sin(f).to(t.dtype)
        rot = f.shape[-1]
        t_rot, t_pass = t[..., :rot], t[..., rot:]
        return torch.cat((t_rot * cos_ + _rotate_half(t_rot) * sin_, t_pass), dim=-1)

    class DotProductAttention(torch.nn.Module):
        def __init__(self, *a, **k):
            raise RuntimeError("Transformer Engine attention is unavailable: use the 'torch' "
                               "attention backend")

    class TransformerEngineBaseModule(torch.nn.Module):
        pass

    te = _module("transformer_engine")
    te.__path__ = []
    pt = _module("transformer_engine.pytorch", RMSNorm=RMSNorm)
    pt.__path__ = []
    te.pytorch = pt
    pt.attention = _module("transformer_engine.pytorch.attention",
                           DotProductAttention=DotProductAttention,
                           apply_rotary_pos_emb=apply_rotary_pos_emb)
    pt.distributed = _module("transformer_engine.pytorch.distributed",
                             get_all_rng_states=lambda: {},
                             graph_safe_rng_available=lambda: False)
    mod = _module("transformer_engine.pytorch.module")
    mod.__path__ = []
    pt.module = mod
    mod.base = _module("transformer_engine.pytorch.module.base",
                       TransformerEngineBaseModule=TransformerEngineBaseModule)


def prepare_imports() -> str:
    """Set up the environment for Fixer's code; returns the attention backend to use."""
    sys.path.insert(0, str(FIXER_SRC))
    _stub_text_encoders()
    _stub_megatron()
    want = os.environ.get("FIXER_ATTENTION", "auto")
    if want != "torch":
        try:
            import transformer_engine.pytorch  # noqa: F401  (loads TE's CUDA libraries)
            return "transformer_engine"
        except Exception as e:  # noqa: BLE001
            print(f"FIXER Transformer Engine unavailable ({type(e).__name__}: {e}); "
                  "using the PyTorch stand-in", flush=True)
            for k in [k for k in sys.modules if k.startswith("transformer_engine")]:
                del sys.modules[k]
    _torch_te_standin()
    return "torch"


# --------------------------------------------------------------------------- model
def load(model_path: str, timestep: int, backend: str):
    import torch
    m = __import__(PIX2PIX)
    # Model files: the build points these at /opt/fixer/models; FIXER_MODELS overrides.
    m.config.dit_path = str(FIXER_MODELS / "base" / "model_fast_tokenizer.pt")
    m.config.tokenizer["vae_pth"] = str(FIXER_MODELS / "base" / "tokenizer_fast.pth")
    m.config.guardrail_config.enabled = False
    if backend == "torch":
        m.config.net.atten_backend = "torch"
    try:
        net = m.Pix2Pix_Turbo(pretrained_path=model_path, timestep=timestep,
                              vae_skip_connection=False, batch_size=1)
    except Exception as e:  # noqa: BLE001
        if "weights_only" not in str(e) and "Unpickl" not in type(e).__name__:
            raise
        # NVIDIA's checkpoint holds optimizer state too; newer torch refuses some of it by
        # default. It's the published nvidia/Fixer file, so allow full loading.
        orig = torch.load
        torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
        try:
            net = m.Pix2Pix_Turbo(pretrained_path=model_path, timestep=timestep,
                                  vae_skip_connection=False, batch_size=1)
        finally:
            torch.load = orig
    net = net.to(device="cuda", dtype=torch.bfloat16)
    net.requires_grad_(False)
    return net


def fix_one(net, img, width: int, height: int):
    """PIL image -> fixed PIL image at the original size (Fixer works at width x height)."""
    import numpy as np
    import torch
    from PIL import Image
    w0, h0 = img.size
    x = img.convert("RGB").resize((width, height), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(x, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    t = (t * 2 - 1).to(device="cuda", dtype=torch.bfloat16)  # Normalize([0.5], [0.5])
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = net(t)
    out = (out.float()[0].clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("Fixer produced non-finite output")
    res = Image.fromarray((out * 255 + 0.5).astype(np.uint8))
    return res.resize((w0, h0), Image.BILINEAR) if (w0, h0) != res.size else res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default=str(FIXER_MODELS / "pretrained" / "pretrained_fixer.pkl"))
    ap.add_argument("--timestep", type=int, default=250)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise SystemExit("FIXER needs a CUDA GPU")
    if torch.cuda.get_device_capability(0)[0] < 8:  # is_bf16_supported() counts emulation
        raise SystemExit(f"FIXER needs native bfloat16 (Ampere or newer); "
                         f"{torch.cuda.get_device_name(0)} doesn't have it")
    t0 = time.time()
    backend = prepare_imports()
    probe_img = Image.fromarray(
        (np.random.default_rng(0).random((args.height, args.width, 3)) * 255).astype("uint8"))
    try:
        net = load(args.model, args.timestep, backend)
        fix_one(net, probe_img, args.width, args.height)  # fail fast, before the batch
    except Exception as e:  # noqa: BLE001
        if backend != "transformer_engine" or isinstance(e, FileNotFoundError):
            raise
        # TE imported but its kernels may not support this GPU: start over in a fresh process
        # on the PyTorch path (reloading in place would re-register Cosmos' config resolvers
        # and keep the failed model's GPU memory)
        print(f"FIXER Transformer Engine path failed ({type(e).__name__}: {str(e)[:300]}); "
              "retrying with PyTorch attention", flush=True)
        os.environ["FIXER_ATTENTION"] = "torch"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    print(f"FIXER loaded in {time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)} "
          f"(attention: {backend})", flush=True)

    if args.probe:
        print(f"FIXER_OK attention={backend} "
              f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1e9:.1f}", flush=True)
        return

    pairs = json.loads(args.pairs.read_text())
    t1 = time.time()
    for k, p in enumerate(pairs):
        out = fix_one(net, Image.open(p["input"]), args.width, args.height)
        out.save(p["output"])
        if (k + 1) % 50 == 0 or k + 1 == len(pairs):
            print(f"FIXER {k + 1}/{len(pairs)} images ({(k + 1) / (time.time() - t1):.1f}/s)",
                  flush=True)
    print(json.dumps({"fixer_done": len(pairs), "attention": backend,
                      "seconds": round(time.time() - t0, 1),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}),
          flush=True)


if __name__ == "__main__":
    main()
