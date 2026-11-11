# 360 video -> Gaussian splat pipeline, packaged for Palantir Foundry compute modules.
#
# Build (from the splats folder; must be linux/amd64):
#   Foundry lightweight transform (T4 etc.; the T4 can't run Fixer, so leave it out):
#     docker build --platform linux/amd64 --build-arg MAX_JOBS=4 --build-arg APP_UID=5001 --build-arg WITH_FIXER=0 -t splat-pipeline:0.2.1 .
#   Foundry compute module:
#     docker build --platform linux/amd64 --build-arg MAX_JOBS=4 -t splat-pipeline:0.2.1 .
#   RTX 5090 / rented GPUs (CUDA 12.8, Blackwell): use pod/build_5090.cmd or .sh, which passes
#     --build-arg CUDA_BASE=12.8.1 --build-arg TORCH_CUDA=cu128
#     --build-arg CUDA_ARCHS="75;89;120" --build-arg TORCH_ARCHS="7.5;8.9;12.0"
#   (sm_120 = RTX 50xx, 8.9 = RTX 40xx / L4, 7.5 = RTX 20xx / T4 so you can test it on a 2080).
# Best built on an x86 PC (Windows + Docker Desktop/WSL 2): native speed, and you can test
# it on the local NVIDIA GPU. Apple Silicon Macs build via slow emulation and can't test it.
#
# Contains: ffmpeg, COLMAP (built with CUDA), PyTorch, gsplat (CUDA kernels prebuilt),
# the pipeline code and the Foundry compute module SDK.
#
# Defaults: CUDA 12.6 (driver >= 560, or >= 525 via minor-version compatibility), kernels for
# T4 (7.5), A100 (8.0), A10G (8.6), L4 (8.9), H100 (9.0) and RTX 2080 (7.5). These GPUs
# can't run an RTX 50xx (sm_120) build without the matching kernels, and vice versa: build
# the image for the GPUs you'll use. CUDA 12.8 needs driver >= 570 (every 5090 host has it).

ARG CUDA_BASE=12.6.3
FROM nvidia/cuda:${CUDA_BASE}-devel-ubuntu24.04

ARG COLMAP_VERSION=4.2.1
ARG CUDA_ARCHS="75;80;86;89;90"
# Same GPUs in PyTorch's notation (dotted), for gsplat's kernels
ARG TORCH_ARCHS="7.5;8.0;8.6;8.9;9.0"
# PyTorch wheel flavour: cu126 for CUDA 12.6, cu128 for CUDA 12.8 (has RTX 50xx kernels)
ARG TORCH_CUDA=cu126
ARG TORCH_VERSION=2.8.0
ARG GSPLAT_VERSION=v1.5.3
# Parallel compile jobs. Each heavy COLMAP/gsplat file can need 2-3 GB of RAM, so keep
# MAX_JOBS <= (Docker memory in GB) / 3. Lower it if the build dies with "Killed" or
# "cannot allocate memory". On an Apple Silicon Mac (amd64 emulation) use 2.
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- system packages: ffmpeg, python, COLMAP build deps (headless: no Qt/GUI) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build build-essential \
        ffmpeg python3 python3-venv python3-dev \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
        libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
        libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev \
        libopenblas-openmp-dev \
    && mkdir -p /usr/include/opencv4 \
    && rm -rf /var/lib/apt/lists/*

# ---- COLMAP with CUDA ----------------------------------------------------------------
RUN git clone --depth 1 --branch ${COLMAP_VERSION} https://github.com/colmap/colmap.git /tmp/colmap \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DGUI_ENABLED=OFF \
        -DBLA_VENDOR=OpenBLAS \
    && cmake --build /tmp/colmap/build --parallel ${MAX_JOBS} \
    && cmake --install /tmp/colmap/build \
    && rm -rf /tmp/colmap \
    && colmap help | head -n 1

# ---- Python environment ----------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCHS}" \
    MAX_JOBS=${MAX_JOBS}

COPY requirements-container.txt /tmp/requirements-container.txt
RUN pip install --upgrade pip \
    && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    && pip install -r /tmp/requirements-container.txt

# gsplat from source compiles its CUDA kernels now, at build time (no GPU needed for this).
RUN pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_VERSION}" \
    && python -c "import gsplat, importlib.util as u; print('gsplat', gsplat.__version__, 'prebuilt kernels:', u.find_spec('gsplat.csrc') is not None)"

# Depth prior weights baked into the image, so the module needs no Hugging Face access at
# runtime (Depth Anything V2 Small, Apache-2.0, ~100 MB).
# Pillow + torchvision are needed by transformers' image preprocessor. Installed here (not in
# requirements-container.txt) so the cached torch/gsplat layers above aren't rebuilt.
# torchvision 0.23 is the release paired with torch 2.8; --no-deps keeps torch untouched.
ARG TORCHVISION_VERSION=0.23.0
ENV HF_HOME=/opt/hf
RUN pip install pillow \
    && pip install --no-deps torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
    && python -c "import torch, torchvision, PIL; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'pillow', PIL.__version__)" \
    && python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation as M; \
n='depth-anything/Depth-Anything-V2-Small-hf'; AutoImageProcessor.from_pretrained(n); M.from_pretrained(n)" \
    && chmod -R a+rwX /opt/hf

# MapAnything (sfm.mapper: mapanything). Apache-2.0 weights baked in, plus the DINOv2 code
# it pulls via torch.hub, so the module needs no internet at runtime. Several GB; build with
# --build-arg WITH_MAPANYTHING=0 to leave it out (the pipeline then falls back to COLMAP).
ARG WITH_MAPANYTHING=1
ARG MAPANYTHING_REF=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
ENV TORCH_HOME=/opt/torchhub
RUN if [ "$WITH_MAPANYTHING" = "1" ]; then \
        printf "torch==${TORCH_VERSION}+${TORCH_CUDA}\ntorchvision==${TORCHVISION_VERSION}+${TORCH_CUDA}\ntorchaudio==${TORCH_VERSION}+${TORCH_CUDA}\n" > /tmp/torch-pins.txt \
        && pip install -c /tmp/torch-pins.txt --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
            "mapanything @ git+https://github.com/facebookresearch/map-anything.git@${MAPANYTHING_REF}" \
        && python -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything-apache')" \
        && mkdir -p /opt/torchhub && chmod -R a+rwX /opt/hf /opt/torchhub ; \
    fi
# NVIDIA Fixer: AI fill-in for the refine step (fixer_refine.py). Its own venv in /opt/fixer
# sharing torch with /opt/venv, Transformer Engine compiled for CUDA_ARCHS, ~5.5 GB of weights.
# Needs a bfloat16 GPU at run time (Ampere or newer: not the T4). --build-arg WITH_FIXER=0 skips.
# Weights: NVIDIA Open Model License (commercial use allowed). Code: Apache-2.0.
ARG WITH_FIXER=1
ARG FIXER_REF=b39dfcaf4eeec90dc943b057ff368c16252c6c6e
COPY pod/install_fixer.sh /tmp/install_fixer.sh
RUN if [ "$WITH_FIXER" = "1" ]; then \
        FIXER_REF=${FIXER_REF} CUDA_ARCHS="${CUDA_ARCHS}" MAX_JOBS=${MAX_JOBS} bash /tmp/install_fixer.sh ; \
    fi && rm -f /tmp/install_fixer.sh
ENV HF_HUB_OFFLINE=1
# Fail the build (not a 2-hour GPU run) if anything above replaced the PyTorch stack, or the
# wheels lack kernels for a GPU this image is built for.
RUN python -c "import torch, torchvision, torchvision.ops, gsplat; from transformers import AutoImageProcessor, AutoModelForDepthEstimation as M; n='depth-anything/Depth-Anything-V2-Small-hf'; AutoImageProcessor.from_pretrained(n); M.from_pretrained(n); \
assert torch.__version__ == '${TORCH_VERSION}+${TORCH_CUDA}', torch.__version__; \
have = [(int(s[3:-1]), int(s[-1])) for s in torch._C._cuda_getArchFlags().split()]; \
want = [tuple(map(int, a.split('.'))) for a in '${TORCH_ARCHS}'.split(';')]; \
missing = [w for w in want if not any(h[0] == w[0] and h[1] <= w[1] for h in have)]; \
assert not missing, f'torch {torch.__version__} has no kernels for compute capability {missing}'; \
print('OK torch', torch.__version__, 'torchvision', torchvision.__version__, 'kernels', torch._C._cuda_getArchFlags())"

# ---- app ------------------------------------------------------------------------------
WORKDIR /opt/app
COPY splat_pipeline.py train_gsplat.py depth_prior.py mapanything_init.py foundry_app.py splat_stream.py selftest.py naming.py scene_fill.py scene_coverage.py appearance.py scaffold.py scaffold_agent.py scaffold_prompt.md retrain.py fixer_fix.py fixer_refine.py config.yaml README.md /opt/app/
RUN if [ -x /opt/fixer/bin/python ]; then FIXER_ATTENTION=torch /opt/fixer/bin/python -c "import sys; sys.path.insert(0, '/opt/app'); import fixer_fix as f; f.prepare_imports(); __import__(f.PIX2PIX); print('Fixer code imports OK')"; fi

# Foundry requires a non-root, numeric user: compute modules accept any (default 5000);
# lightweight "bring your own container" transforms require exactly 5001, so build those
# with --build-arg APP_UID=5001. /work holds per-video working files (tens of GB at 8K).
ARG APP_UID=5000
# Foundry's container-transform launcher (init.sh) uses curl, sed, coreutils and other basic
# shell tools inside the image to fetch its runtime, so they must be present. Installed late so the expensive
# COLMAP/torch/gsplat layers above stay cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates coreutils sed grep gawk tar gzip unzip findutils procps bash \
    && rm -rf /var/lib/apt/lists/* \
    && curl --version | head -n 1
RUN mkdir -p /opt/torchhub /opt/hf \
    && chmod -R a+rwX /opt/hf /opt/torchhub \
    && useradd --uid ${APP_UID} --create-home splat \
    && mkdir -p /work /input /output \
    && chown -R ${APP_UID}:${APP_UID} /opt/app /work /input /output
ENV HOME=/home/splat \
    TORCH_EXTENSIONS_DIR=/home/splat/.cache/torch_extensions \
    SPLAT_WORK=/work \
    COLMAP_BIN=colmap \
    SPLAT_MODE=auto
USER ${APP_UID}

CMD ["python", "/opt/app/foundry_app.py"]
