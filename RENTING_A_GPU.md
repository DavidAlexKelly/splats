# Running the pipeline on a rented RTX 5090 (RunPod or Vast.ai)

The same pipeline as on Foundry, in your own rented GPU machine ("pod"): upload videos, run
one command, download the `.ply` files, stop the pod. You pay only while it's running
(RTX 5090: roughly $0.50-0.70/hour as of November 2026).

```
your PC ──scp──► pod: /workspace/videos ──splat──► /workspace/splats ──scp──► your PC ──► viewer
```

The Foundry setup keeps working unchanged (the default build is still the T4 one).

---

## 1. One-time: build the image (on your PC)

The 5090 is a Blackwell GPU: it needs CUDA 12.8 and code compiled for it, so it gets its own
image. This one also includes kernels for RTX 40xx and RTX 20xx, so you can rent a 4090 if
5090s are sold out, and test it on your own 2080.

1. Make a free [Docker Hub](https://hub.docker.com) account and a **private** repository,
   e.g. `yourname/splat-pod`.
2. In a terminal (cmd), from the `splats` folder:
   ```
   docker login
   pod\build_5090.cmd yourname/splat-pod 1.0.0
   ```
   The first build takes 1-2 hours (compiling COLMAP and gsplat) and pushes ~20 GB.
   If it dies with "Killed" / out of memory, lower `MAX_JOBS=4` in `pod\build_5090.cmd` to 2.
3. Later code or setting changes (no new dependencies) take a minute or two:
   ```
   pod\update_pod.cmd yourname/splat-pod 1.0.1 1.0.0
   ```
   (new tag, then the base tag from step 2). Put the new tag in your template.

Optional quick test on your own PC (the 2080 is supported by this image):
```
docker run --rm --gpus all --entrypoint "" yourname/splat-pod:1.0.0 python /opt/app/selftest.py --quick
```

### On a Mac (or without Docker): let GitHub build it

Apple Silicon Macs can't build this image natively (it's for Intel/AMD Linux machines), so
GitHub's free Linux machines build it and push it to Docker Hub
(`.github/workflows/build-pod-image.yml`):

1. Docker Hub: private repository `splat-pod` + an access token (Read & Write), as above.
2. GitHub: create a new **private** repository (no README), e.g. `splats`. In Terminal:
   ```
   cd ~/path/to/splats
   git init && git add -A && git commit -m "splat pipeline"
   git branch -M main
   git remote add origin https://github.com/YOURNAME/splats.git
   git push -u origin main
   ```
   (`git` asks to install the Command Line Tools the first time: say yes. For the password,
   use a GitHub personal access token, or install GitHub Desktop and publish the folder.)
3. On GitHub, in the repository: **Settings → Secrets and variables → Actions**
   - *Secrets* tab: `DOCKERHUB_USERNAME` = your Docker Hub username,
     `DOCKERHUB_TOKEN` = the access token
   - *Variables* tab: `IMAGE` = `yourname/splat-pod`
4. **Actions → Build pod image → Run workflow** (tag `1.0.0`). It takes about 3-5 hours; you
   get an email if it fails. Result: `yourname/splat-pod:1.0.0` on Docker Hub.
5. Code changes later: commit + push, then run the workflow again with **rebuild base**
   unticked, a new tag (e.g. `1.0.1`) and base tag `1.0.0`: a few minutes.

The default GPU list is `89;120` (RTX 50xx and 40xx); add `86` for 30xx / A10 rentals.

## 2. One-time: an SSH key

Windows (cmd) or Mac (Terminal): `ssh-keygen -t ed25519` (press Enter for the defaults). Your
public key is the text in `C:\Users\<you>\.ssh\id_ed25519.pub` (Windows) or
`~/.ssh/id_ed25519.pub` (Mac: `pbcopy < ~/.ssh/id_ed25519.pub` copies it); you'll paste it
into RunPod / Vast.ai.

## 3a. RunPod

One-time setup:
1. **Settings → SSH Public Keys**: paste your public key.
2. **Settings → Container Registry Auth**: add your Docker Hub username + an access token
   (Docker Hub → Account settings → Personal access tokens), so RunPod can pull your private image.
3. **Templates → New Template** (type Pod):
   - Container image: `yourname/splat-pod:1.0.0`, registry credentials: the ones from step 2
   - Container disk: 30 GB. Volume disk: 200 GB, mount path `/workspace`
   - Expose TCP ports: `22`
   - Leave the start command empty (the image starts SSH and waits)

Each time:
1. **Pods → Deploy**: pick **RTX 5090**, **On-Demand** (not Spot/Interruptible: an interrupted
   pod loses the run), choose your template, deploy.
2. When it's running: **Connect → SSH over exposed TCP** shows something like
   `ssh root@203.0.113.7 -p 40123 -i ~/.ssh/id_ed25519`. Use those numbers below.

## 3b. Vast.ai (cheaper, a marketplace of other people's machines)

One-time setup:
1. **Account → Keys**: paste your SSH public key.
2. **Templates → New**: image `yourname/splat-pod:1.0.0`, Docker login with your Docker Hub
   username + access token, launch mode **SSH**, disk 200 GB, on-start script:
   ```
   mkdir -p /workspace/videos /workspace/splats /workspace/work; env >> /etc/environment
   ```

Each time: **Search**, filter on RTX 5090 (or 4090), prefer high reliability and fast
internet (upload speed matters for 8K videos), rent with your template. **Instances →
Connect** gives the SSH command.

## 3c. Claude for the scene scaffold (recommended)

Claude reviews each scene's surfaces before training (see README, "Scene scaffold"). It
needs an Anthropic API key: create one at console.anthropic.com (API keys), add some credit,
and either
- add it to your RunPod template as an environment variable `ANTHROPIC_API_KEY`
  (Edit Template → Environment Variables), then start a new pod; or
- on a running pod: `echo 'sk-ant-...' > /workspace/.anthropic_key` (kept on the volume).

Each scene costs roughly $0.50-3, capped at `scaffold_max_cost` (default $3); the decision is
saved, so re-runs and `--retrain` don't pay again. Without a key everything still works with
automatic rules (the start-up banner and `splat` say which one is used).

Cheapest way to try it on a result you already have: `splat --scaffold-only patio_v2`
(a few minutes, no training), look at `patio_v2_scaffold/scaffold_review.jpg`, then
`splat --retrain patio_v2 --no-fixer` trains with it (about 45 min instead of 2 hours).
Note: results made before this update have no saved scaffold, and `--retrain` needs the
original video in /workspace/videos (the training images are re-cut from it).

## 4. Upload, run, download

From cmd on your PC, or Terminal on a Mac (replace the IP/port and paths):
```
scp -P 40123 D:\drone\patio.mp4 root@203.0.113.7:/workspace/videos/
ssh -p 40123 root@203.0.113.7
```
On the pod:
```
tmux new -s run              # keeps running if your connection drops (reattach: tmux attach -t run)
splat --selftest             # first time on a new image: GPU, COLMAP, gsplat and Fixer (~5 min)
splat                        # every video in /workspace/videos: splat, then Fixer
```
Useful variations:
```
splat patio.mp4                              # one video
splat --no-fixer                             # real data only, no AI fill-in
splat --fixer-only patio_v2                  # refine an existing result again
splat --scaffold-only patio_v2               # just Claude's scaffold (minutes, ~$1): check it first
splat --retrain patio_v2 --no-fixer          # train again, skipping the ~50 min camera solve
splat --retrain patio_v2 --decision none     # ... with Claude deciding the scaffold afresh
splat --compare patio_v2 patio_v3            # side-by-side fly-through: _compare/*.mp4
splat --rerun                                # process again with unchanged settings
splat --overrides '{"train":{"custom":{"cap_max":4000000}}}'   # fewer Gaussians if memory runs out
splat --fixer-overrides '{"fixer":{"rounds":2}}'
splat --stop-when-done                       # stop the pod at the end, even on failure
```
Results, in `/workspace/splats` (never overwritten; re-runs become `patio_v2`, `_v3`...):
```
patio/patio.ply              the splat (real data)
patio/eval/*.jpg             photo vs splat comparisons; train_stats.json, pipeline.log, scene/
patio/flythrough.mp4         video along a slightly shifted path: the quickest way to judge popping
patio/scaffold_review.jpg    Claude's scaffold over photos + from above (solid green, glass blue, soft yellow)
patio/scaffold_agent_log.md  what Claude did and why, and what it cost
patio_fixer/patio_fixer.ply  the AI-filled version; fixer_stats.json, fixer_preview/*.jpg
_logs/                       full logs of every run
```
Download (from cmd on your PC):
```
scp -P 40123 -r root@203.0.113.7:/workspace/splats D:\drone\splats
```
Then open them in the viewer: run it without a `.env` (local mode) and press **Folder…** to
pick `D:\drone\splats`; or drag a `.ply` into [SuperSplat](https://superspl.at/editor).

## 5. Stop paying

- **Stop** the pod: no GPU charge; the volume (your videos and results) is kept for a small
  storage fee. A stopped RunPod pod may not get the same GPU back when restarted.
- **Terminate** it: everything is deleted, no further charges. Download first.
- `splat --stop-when-done` stops it for you. If it says it couldn't, stop it on the website.

Hands-off (RunPod): with your videos already on a network volume, set the template's
environment variables `AUTO_RUN=1` and `POD_RUN_ARGS=--stop-when-done`: the pod processes
everything on start and stops itself. Log: `/workspace/splats/_logs/auto_run.log`.

## 6. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `no kernel image is available for execution on the device` | Wrong image for the GPU (e.g. the Foundry T4 image on a 5090). Use the image from `build_5090`. |
| `CUDA driver version is insufficient` | That host's NVIDIA driver is older than 570. Pick another machine. |
| `Permission denied (publickey)` | The SSH key wasn't in your account settings when the pod started: add it, restart the pod. |
| `No space left on device` | Bigger volume (an 8K video can need 30-60 GB of working space). |
| Pod can't pull the image | Registry credentials missing/wrong in the template, or the tag doesn't exist. |
| A run stopped when you closed the window | Run it inside `tmux` (see step 4). |

Defaults that suit a rented GPU: the number of Gaussians follows the GPU's memory (5M on a
5090, 4M on a 4090). Camera solving uses COLMAP's incremental mapper (about 50 minutes for
patio, on the processor; the GPU idles meanwhile). The global mapper is available with
`--overrides '{"sfm":{"mapper":"auto","auto_global_timeout_min":60}}'`, but on patio it was
no quicker.

The `_fixer` results come from NVIDIA Fixer (weights under the NVIDIA Open Model License:
commercial use allowed) and contain AI-generated content; the real-only splat is always
kept next to them. Fixer needs a bfloat16 GPU (RTX 30xx or newer, A100...).
