@echo off
REM Code or config change only (no new dependencies): rebuild just the pod layer and push.
REM     pod\update_pod.cmd yourdockerhubname/splat-pod 1.0.1 1.0.0
REM     (1.0.1 = the new tag, 1.0.0 = the base image tag you built with build_5090)
setlocal
set REPO=%~1
set TAG=%~2
set BASETAG=%~3
if "%BASETAG%"=="" (echo Usage: pod\update_pod.cmd yourdockerhubname/splat-pod NEWTAG BASETAG & exit /b 1)
docker build --platform linux/amd64 -f Dockerfile.pod --build-arg BASE=splat-5090-base:%BASETAG% ^
  -t %REPO%:%TAG% . || exit /b 1
docker push %REPO%:%TAG% || exit /b 1
echo Done: %REPO%:%TAG%  (update the image tag in your template)
