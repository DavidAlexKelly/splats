#!/bin/bash
# Pod entry point: folders, SSH, an optional automatic run, then stay up so you can connect.
set -u
mkdir -p /workspace/videos /workspace/splats/_logs /workspace/work

# SSH. RunPod passes the public key from your account settings in $PUBLIC_KEY (Vast.ai runs
# its own SSH setup in "SSH" launch mode, so this part is skipped there).
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null \
        || echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1
    /usr/sbin/sshd && echo "SSH server started"
fi
# Make the image's environment visible to SSH sessions too (sshd starts them clean).
env | grep -E '^(PATH|HF_|TORCH_|SPLAT_|COLMAP_BIN|RUNPOD_|CONTAINER_|NVIDIA_|LD_LIBRARY_PATH)' \
    | sed 's/^/export /; s/=\(.*\)$/="\1"/' > /etc/profile.d/splat_env.sh

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "WARNING: no GPU visible (nvidia-smi failed)"
cat <<'MSG'
------------------------------------------------------------------------------
 Splat pod ready.  Upload videos to /workspace/videos, then run:
     splat --selftest          (first time: checks GPU, COLMAP, gsplat, Difix)
     splat                     (all videos: splat + Difix -> /workspace/splats)
     splat --stop-when-done    (same, then stops the pod)
 Use tmux so a dropped connection doesn't stop the run:  tmux new -s run
------------------------------------------------------------------------------
MSG

# Hands-off mode: set AUTO_RUN=1 (and optionally POD_RUN_ARGS="--stop-when-done") in the
# pod's environment variables and it processes everything in /workspace/videos on start.
if [ "${AUTO_RUN:-0}" = "1" ]; then
    # shellcheck disable=SC2086
    splat ${POD_RUN_ARGS:-} 2>&1 | tee -a /workspace/splats/_logs/auto_run.log
fi
exec sleep infinity
