#!/bin/bash
# Linux/macOS/WSL version of build_5090.cmd. Run from the splats folder:
#   pod/build_5090.sh yourdockerhubname/splat-pod 1.0.0          # full build + push
#   pod/build_5090.sh yourdockerhubname/splat-pod 1.0.1 1.0.0    # code change: pod layer only
set -euo pipefail
REPO=${1:?usage: pod/build_5090.sh yourdockerhubname/splat-pod TAG [BASETAG]}
TAG=${2:-1.0.0}
BASETAG=${3:-}
if [ -z "$BASETAG" ]; then
    BASETAG=$TAG
    docker build --platform linux/amd64 --build-arg MAX_JOBS="${MAX_JOBS:-4}" \
        --build-arg CUDA_BASE=12.8.1 --build-arg TORCH_CUDA=cu128 \
        --build-arg "CUDA_ARCHS=75;89;120" --build-arg "TORCH_ARCHS=7.5;8.9;12.0" \
        -t "splat-5090-base:$BASETAG" .
fi
docker build --platform linux/amd64 -f Dockerfile.pod --build-arg BASE="splat-5090-base:$BASETAG" \
    -t "$REPO:$TAG" .
docker push "$REPO:$TAG"
echo "Done: $REPO:$TAG"
