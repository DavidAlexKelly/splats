# 360 video -> Gaussian splat pipeline, packaged for Palantir Foundry compute modules.
#
# Build (from the splats folder; must be linux/amd64 for Foundry):
#   docker build --platform linux/amd64 --build-arg MAX_JOBS=4 -t splat-pipeline:0.1.0 .
# Best built on an x86 PC (Windows + Docker Desktop/WSL 2): native speed, and you can test
# it on the local NVIDIA GPU. Apple Silicon Macs build via slow emulation and can't test it.
#
# Contains: ffmpeg, COLMAP (built with CUDA), PyTorch, gsplat (CUDA kernels prebuilt),
# the pipeline code and the Foundry compute module SDK.
#
# CUDA 12.6 needs an NVIDIA driver >= 560 on the host (or >= 525 via CUDA minor-version
# compatibility). Kernels are built for T4 (7.5), A100 (8.0), A10G (8.6), L4 (8.9),
# H100 (9.0) and RTX 2080 (7.5) so the same image runs on all of them.

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG COLMAP_VERSION=4.2.1
ARG CUDA_ARCHS="75;80;86;89;90"
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
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    MAX_JOBS=${MAX_JOBS}

COPY requirements-container.txt /tmp/requirements-container.txt
RUN pip install --upgrade pip \
    && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu126 \
    && pip install -r /tmp/requirements-container.txt

# gsplat from source compiles its CUDA kernels now, at build time (no GPU needed for this).
RUN pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_VERSION}" \
    && python -c "import gsplat, importlib.util as u; print('gsplat', gsplat.__version__, 'prebuilt kernels:', u.find_spec('gsplat.csrc') is not None)"

# Depth prior weights baked into the image, so the module needs no Hugging Face access at
# runtime (Depth Anything V2 Small, Apache-2.0, ~100 MB).
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation as M; \
n='depth-anything/Depth-Anything-V2-Small-hf'; AutoImageProcessor.from_pretrained(n); M.from_pretrained(n)" \
    && chmod -R a+rX /opt/hf

# MapAnything (sfm.mapper: mapanything). Apache-2.0 weights baked in, plus the DINOv2 code
# it pulls via torch.hub, so the module needs no internet at runtime. Several GB; build with
# --build-arg WITH_MAPANYTHING=0 to leave it out (the pipeline then falls back to COLMAP).
ARG WITH_MAPANYTHING=1
ARG MAPANYTHING_REF=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
ENV TORCH_HOME=/opt/torchhub
RUN if [ "$WITH_MAPANYTHING" = "1" ]; then \
        pip install "mapanything @ git+https://github.com/facebookresearch/map-anything.git@${MAPANYTHING_REF}" \
        && python -c "from mapanything.models import MapAnything; MapAnything.from_pretrained('facebook/map-anything-apache')" \
        && mkdir -p /opt/torchhub && chmod -R a+rX /opt/hf /opt/torchhub ; \
    fi
ENV HF_HUB_OFFLINE=1

# ---- app ------------------------------------------------------------------------------
WORKDIR /opt/app
COPY splat_pipeline.py train_gsplat.py depth_prior.py mapanything_init.py foundry_app.py splat_stream.py config.yaml README.md /opt/app/

# Foundry requires a non-root, numeric user. /work holds per-video working files: mount a
# volume there (tens of GB per video at 8K).
RUN useradd --uid 5000 --create-home splat \
    && mkdir -p /work /input /output \
    && chown -R 5000:5000 /opt/app /work /input /output
ENV HOME=/home/splat \
    TORCH_EXTENSIONS_DIR=/home/splat/.cache/torch_extensions \
    SPLAT_WORK=/work \
    COLMAP_BIN=colmap \
    SPLAT_MODE=auto
USER 5000

CMD ["python", "/opt/app/foundry_app.py"]
