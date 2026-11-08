@echo off
REM Builds the RTX 5090 image and pushes it to your registry. Run from the splats folder:
REM     pod\build_5090.cmd yourdockerhubname/splat-pod 1.0.0
REM First run: 1-2 hours (COLMAP + gsplat compile for 3 GPU generations). Later code changes:
REM use pod\update_pod.cmd instead (a minute or two).
setlocal
set REPO=%~1
set TAG=%~2
if "%REPO%"=="" (echo Usage: pod\build_5090.cmd yourdockerhubname/splat-pod 1.0.0 & exit /b 1)
if "%TAG%"=="" set TAG=1.0.0

echo === 1/3 base image (CUDA 12.8, kernels for RTX 50xx / 40xx / 20xx) ===
docker build --platform linux/amd64 --build-arg MAX_JOBS=4 ^
  --build-arg CUDA_BASE=12.8.1 --build-arg TORCH_CUDA=cu128 ^
  --build-arg "CUDA_ARCHS=75;89;120" --build-arg "TORCH_ARCHS=7.5;8.9;12.0" ^
  -t splat-5090-base:%TAG% . || exit /b 1

echo === 2/3 pod layer (SSH, runner, /workspace) ===
docker build --platform linux/amd64 -f Dockerfile.pod --build-arg BASE=splat-5090-base:%TAG% ^
  -t %REPO%:%TAG% . || exit /b 1

echo === 3/3 push ===
docker push %REPO%:%TAG% || exit /b 1
echo Done: %REPO%:%TAG%
echo Use this as the container image in your RunPod template / Vast.ai template.
