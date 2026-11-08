#!/bin/bash
# Installs NVIDIA Fixer (github.com/nv-tlabs/Fixer + huggingface.co/nvidia/Fixer) into
# /opt/fixer for fixer_refine.py. Run during the image build (see Dockerfile), as root.
#
#   /opt/fixer/bin/python   its own venv; torch etc. come from /opt/venv through a .pth file
#                           (appended after the venv's own packages, so Fixer's pinned
#                           diffusers/transformers win inside this venv only)
#   /opt/fixer/src          Fixer's model code (Apache-2.0), model paths pointed at /opt/fixer/models
#   /opt/fixer/models       weights, ~5.5 GB (NVIDIA Open Model License: commercial use allowed)
#
# Only what single-image inference needs is installed: cosmos-predict2 without its training
# stack. fixer_fix.py supplies single-GPU stand-ins for megatron-core and the unused text
# encoders. Transformer Engine (fast kernels) is compiled here; if that fails the build goes
# on and Fixer uses its PyTorch path instead (same maths, a bit slower).
set -euo pipefail

FIXER_REF=${FIXER_REF:-b39dfcaf4eeec90dc943b057ff368c16252c6c6e}
TE_VERSION=${TE_VERSION:-2.8.0}
WITH_TE=${WITH_TE:-1}
ARCHS=${CUDA_ARCHS:-"89;120"}
JOBS=${MAX_JOBS:-4}

echo "=== Fixer: venv"
/opt/venv/bin/python -m venv /opt/fixer
MAIN_SP=$(/opt/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
FX_SP=$(/opt/fixer/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$MAIN_SP" > "$FX_SP/zz_main_venv.pth"
PIP="/opt/fixer/bin/pip install --no-cache-dir"

echo "=== Fixer: Python packages"
$PIP --no-deps "cosmos-predict2==1.0.9"
$PIP "diffusers==0.33.1" "transformers==4.51.3" "huggingface-hub>=0.30,<1.0" "tokenizers<0.22" \
    attrs hydra-core omegaconf loguru termcolor einops iopath fvcore natsort imageio av pandas \
    requests safetensors nvidia-ml-py

if [ "$WITH_TE" = "1" ]; then
    echo "=== Fixer: Transformer Engine $TE_VERSION (compiles its PyTorch bindings; 10-40 min)"
    $PIP setuptools wheel pybind11 ninja packaging
    # TE supports compute capability 8.0+ only (Fixer needs bfloat16, i.e. 8.0+, anyway)
    TE_ARCHS=$(echo "$ARCHS" | tr ';' '\n' | awk '$1 >= 80' | paste -sd';')
    TE_ARCHS=${TE_ARCHS:-"89;120"}
    # cuDNN comes from PyTorch's pip wheel: headers there, and an unversioned .so to link to
    CUDNN_DIR="$MAIN_SP/nvidia/cudnn"
    CUDNN_INC="$CUDNN_DIR/include"
    for lib in "$CUDNN_DIR"/lib/libcudnn*.so.9; do
        [ -e "$lib" ] && ln -sf "$(basename "$lib")" "${lib%.9}"
    done
    if ( export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS="$TE_ARCHS" MAX_JOBS="$JOBS" \
             CUDNN_PATH="$CUDNN_DIR" CUDNN_HOME="$CUDNN_DIR" \
             LIBRARY_PATH="$CUDNN_DIR/lib:${LIBRARY_PATH:-}" \
             CPLUS_INCLUDE_PATH="$CUDNN_INC:${CPLUS_INCLUDE_PATH:-}" \
             C_INCLUDE_PATH="$CUDNN_INC:${C_INCLUDE_PATH:-}"; \
         $PIP --no-build-isolation "transformer_engine[pytorch]==$TE_VERSION" ); then
        echo "Transformer Engine installed"
    else
        echo "WARNING: Transformer Engine didn't build; Fixer will use its PyTorch path"
        /opt/fixer/bin/pip uninstall -y transformer_engine transformer_engine_torch \
            transformer_engine_cu12 >/dev/null 2>&1 || true
    fi
fi

echo "=== Fixer: model code ($FIXER_REF)"
rm -rf /tmp/fixer
git clone -q https://github.com/nv-tlabs/Fixer.git /tmp/fixer
git -C /tmp/fixer checkout -q "$FIXER_REF"
mkdir -p /opt/fixer/src
cp /tmp/fixer/src/pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py /tmp/fixer/src/model.py \
   /tmp/fixer/LICENSE.txt /tmp/fixer/THIRD_PARTY_LICENSE.txt /opt/fixer/src/
sed -i 's#/work/models/#/opt/fixer/models/#g' \
    /opt/fixer/src/pix2pix_turbo_nocond_cosmos_base_faster_tokenizer.py
rm -rf /tmp/fixer

echo "=== Fixer: weights (~5.5 GB)"
HF_HUB_OFFLINE=0 /opt/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Fixer', local_dir='/opt/fixer/models',
                  allow_patterns=['base/*', 'pretrained/*', 'README.md'])"
for f in base/model_fast_tokenizer.pt base/tokenizer_fast.pth pretrained/pretrained_fixer.pkl; do
    test -s "/opt/fixer/models/$f" || { echo "missing /opt/fixer/models/$f"; exit 1; }
done
rm -rf /opt/fixer/models/.cache

chmod -R a+rX /opt/fixer
du -sh /opt/fixer
