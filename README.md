# 360 video → Gaussian splat pipeline

Turns stitched 360° video (e.g. DJI Avata 360) into a Gaussian splat `.ply`, end to end:

```
extract → select → slice     → sfm          → gate → train → export
ffmpeg    sharpest  360→cube    COLMAP with    quality  gsplat + depth   out/<name>.ply
          frames    map (6      a camera rig,  check    prior trainer
                    views)      or MapAnything
```

```
splats/
├── foundry_app.py          # one entrypoint: detects PC / Docker / Foundry and runs the right mode
├── splat_pipeline.py       # orchestrator (resumable stages), incl. the MapAnything+COLMAP hybrid
├── train_gsplat.py         # our trainer (PyTorch + gsplat) + --check / --inspect / --depth-preview
├── depth_prior.py          # monocular depth prior (selective, aligned to SfM)
├── mapanything_init.py     # metric pose initialisation for sfm.mapper: mapanything
├── naming.py               # view layouts (cube map / custom), image naming, COLMAP rig config
├── scene_fill.py           # depth seeding, sky shell, floater cleanup (filling gaps, real data only)
├── appearance.py           # per-photo bilateral colour grids (off by default)
├── scaffold.py             # scene surfaces: candidates, photo checks, outlines, training rules
├── scaffold_agent.py       # Claude reviews the surfaces with tools and sets the rules
├── scaffold_prompt.md      # Claude's instructions for that (override: /workspace/scaffold_prompt.md)
├── retrain.py              # train again from a result's scene bundle (no camera solve)
├── fixer_refine.py         # AI fill-in on a finished splat with NVIDIA Fixer
├── fixer_fix.py            # runs Fixer itself (in /opt/fixer, the image's Fixer environment)
├── pod_run.py              # one command on a rented GPU: videos in, splats (+ Fixer) out
├── pod/                    # rented-GPU image: build scripts, start script, install_fixer.sh
├── selftest.py             # checks every stage in minutes (see "Self-test")
├── splat_stream.py         # file <-> stream-record chunking for Foundry stream outputs
├── foundry_transform/
│   ├── splat_transform.py      # Foundry lightweight transform (recommended Foundry route)
│   ├── reconstruct_splats.py   # compute-module route: rebuild files from the stream
│   └── splat_stream_schema.json
├── config.yaml             # every setting lives here
├── requirements.txt        # PC install
├── requirements-container.txt, Dockerfile, Dockerfile.quick, Dockerfile.pod, .dockerignore
├── .github/workflows/      # builds the rented-GPU image on GitHub (no local Docker needed)
├── FOUNDRY.md              # Foundry deployment guide
├── RENTING_A_GPU.md        # RunPod / Vast.ai guide
├── tools/                  # COLMAP Windows release (COLMAP.bat, bin/, lib/) - you add this
├── raw/                    # your equirectangular videos
├── work/                   # per-video working folders (incl. pipeline.log)
└── out/                    # final splats (.ply) + run summaries
```

---

## 1. Capturing and exporting

- Fly slowly and **move through the space** (forwards/back, up/down, side to side).
  Hovering in one spot gives no parallax and the quality gate will reject it.
  60 s for a room or patio is plenty.
- Indoors: prop wash moves curtains/plants (→ floaters); lock exposure if you can.
- **Export from DJI Studio as stitched equirectangular**, not the raw `.OSV`:
  drag one clip to the timeline → set D-Log M colour recovery (only if shot in D-Log M)
  → perspective *Free View* → Export → *Panoramic Video*.
  - Resolution **8K 7680×3840** (must be 2:1), bitrate at max (480 Mbps),
    SDR Rec709, 10-bit fine.
  - Frame rate: pick the rate you recorded at if offered (avoids blended frames).
  - Noise reduction: on for dim scenes, off for well-lit ones (it can smooth away texture).
  - Turn on horizon levelling if available, so "up" in the video is real up.

---

## 2. Windows setup (NVIDIA GPU)

Tested on Windows 10 + RTX 2080 (8 GB). Every problem we hit is listed in
**Troubleshooting** below with its fix.

### 2.1 Put the project in a path with no spaces

e.g. `C:\Users\you\Documents\MyDocuments\splats` or simply `C:\splats`.
Spaces in the path ("My Documents") break COLMAP's launcher and gsplat's CUDA build.

### 2.2 ffmpeg

```
winget install Gyan.FFmpeg
```

Then **close and reopen the terminal** and check `ffprobe -version`. If it's still not
found, see Troubleshooting → *ffprobe not recognised*.

### 2.3 NVIDIA driver

Install the latest Game Ready or Studio driver and reboot. `nvidia-smi` should show
driver **580+ / CUDA 13.x** (needed by current COLMAP CUDA builds).

### 2.4 COLMAP

Download the **CUDA** Windows release from github.com/colmap/colmap/releases and unzip
it into `tools\` so you have `tools\COLMAP.bat`, `tools\bin\colmap.exe`, `tools\lib\`.
In `config.yaml`:

```yaml
sfm:
  colmap_bin: tools/COLMAP.bat   # the pipeline calls bin\colmap.exe directly
  use_gpu: true
```

Quick test: `tools\bin\colmap.exe help` prints a version with "with CUDA".

### 2.5 Compilers for gsplat (one-off)

1. **Visual Studio 2022 Build Tools** with the C++ workload:
   ```
   winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
   ```
2. **CUDA Toolkit 12.8** from NVIDIA's CUDA Toolkit Archive (Windows → x86_64 → exe local),
   default options. It must match the `cu128` PyTorch below. Check: `nvcc --version` → `release 12.8`.

### 2.6 Open the right terminal

From the Start menu open **x64 Native Tools Command Prompt for VS 2022**
(the name must start with **x64**). Check:

```
cl
```

The first line must end with **`for x64`**. If it says `for x86`, run:

```
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Use this prompt for installing gsplat and the first `--check`.

### 2.7 Python virtual environment

We use the **Python 3.11** that's already installed (3.12 wasn't). Don't install into the
system Python: it has a CPU-only PyTorch that will silently be used instead.

```
cd C:\Users\you\Documents\MyDocuments\splats
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The prompt must now start with `(.venv)`. **Every new terminal needs
`.venv\Scripts\activate` again.**

> **Always use `python`, never `py`, once the venv is active.** `py` ignores the venv and
> runs the system Python (CPU-only torch → "CUDA not available").

### 2.8 PyTorch (CUDA build, version 2.8.0)

```
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Must print `2.8.0+cu128 True`. (torch 2.11 fails to build gsplat on Windows, see below.)

### 2.9 gsplat, plus the Windows patch

```
pip install packaging
pip install gsplat
```

gsplat 1.5.3 passes a Linux-only compiler flag that Microsoft's `cl` rejects. Remove it:

```
python -c "import pathlib;p=pathlib.Path(r'.venv\Lib\site-packages\gsplat\cuda\_backend.py');s=p.read_text();print('found',s.count('-Wno-attributes'));p.write_text(s.replace('\"-Wno-attributes\"','').replace(\"'-Wno-attributes'\",''))"
findstr /s /n /c:"Wno-attributes" .venv\Lib\site-packages\gsplat\*.py
```

The `findstr` should now print nothing. **Reapply this after any gsplat reinstall/upgrade.**

### 2.10 Build and test gsplat

```
rmdir /s /q "%LOCALAPPDATA%\torch_extensions"
python train_gsplat.py --check > build.log 2>&1
findstr /i /c:"error" /c:"OK:" build.log
```

The build takes 5–15 minutes with no visible progress (output goes to `build.log`).
Success is:

```
OK: rendered and back-propagated through gsplat on the GPU
```

After this the compiled kernels are cached and any terminal (with the venv active) works.

---

## 3. Running

Close GPU-hungry apps first (Ollama, DJI Studio, Wallpaper Engine, games).

```
python splat_pipeline.py raw/patio.mp4 --config config.yaml --until-stage gate   # check capture first
python splat_pipeline.py raw/patio.mp4 --config config.yaml                      # full run
```

Finished stages are skipped on re-runs. Other useful commands:

```
python splat_pipeline.py raw/ --config config.yaml                          # every video in a folder
python splat_pipeline.py raw/patio.mp4 --config config.yaml --from-stage train   # retrain only
python train_gsplat.py --inspect --data work/patio --config config.yaml     # is the geometry sound?
```

To **retrain from scratch** with new settings: `rmdir /s /q work\patio\train` first
(otherwise it resumes from the last checkpoint).

### What to expect (patio, 60 s clip, RTX 2080)

| Stage | Time |
|---|---|
| extract + select + slice | ~3 min |
| feature extraction (GPU) | ~3 min |
| sequential matching (GPU) | ~8 min |
| mapper: `colmap` incremental | ~2 h 20 min |
| mapper: `colmap_global` | minutes, but hung on the 214-frame re-run (`auto`, the default, tries it with a 20 min limit, then falls back to incremental) |
| mapper: `mapanything` (hybrid) | MapAnything minutes + COLMAP triangulation/BA (expected well under the incremental mapper) |
| training, 1M Gaussians, 30k steps | ~48 min, 2.2 GB VRAM |
| training, 3M Gaussians | ~1.5–2 h (estimate) |

Result: 1,562 / 1,859 images registered (the missing ones were mostly straight-up sky views),
training PSNR ≈ 26.

### Reading the outputs

- Final splat: `out\<name>.ply`. Drag into **superspl.at/editor** to view, crop floaters,
  rescale and export compressed formats.
- Everything is logged to `work\<name>\pipeline.log`, even if you close the terminal:
  ```
  findstr /c:"psnr" work\patio\pipeline.log                       # training progress
  findstr /c:"Aligned" /c:"disagree" work\patio\pipeline.log       # orientation check
  powershell -c "Get-Content work\patio\pipeline.log -Tail 20"      # end of run
  powershell -c "Get-Content work\patio\pipeline.log -Tail 20 -Wait"  # watch live
  type work\patio\train\train_stats.json
  ```

### Judging quality honestly

The `psnr` in the progress lines is measured on training photos, so it flatters the model.
Every 20th image is held out (`test_every`) and scored every 5,000 steps:

```
findstr /c:"EVAL" work\patio\pipeline.log
```

Compare **held-out PSNR** between runs to know whether a change really helped. At the end,
`work\<name>\train\eval\` contains side-by-side *photo | splat* JPGs of unseen views,
the quickest way to see where blur is.

`exposure: true` learns a small colour/exposure correction per photo, so auto-exposure
shifts (sky ↔ shade ↔ glass) don't get baked into the splat as blur and floaters.

### How the 360 frames are used: cube map + camera rig

Each 360 frame is cut into normal camera views that COLMAP and the trainer understand,
saved as `images/v00/f00012.jpg` (view 0 of frame 12), `images/v01/f00012.jpg`, ...

- **Cube map (default, `slice.layout: cubemap`)**: 6 non-overlapping faces
  (front, right, back, left, up, down). Every part of the sphere appears exactly once, so the
  trainer sees no duplicated pixels, and there are ~3x fewer images than the older 17
  overlapping views. Drop faces with `slice.cubemap_faces` (e.g. `up` for open sky).
  `slice.layout: custom` uses the overlapping `slice.views` list instead; then each pixel is
  "owned" by its nearest view for SfM so features aren't extracted several times
  (`sfm_masks/`), while training still uses every pixel.
- **Camera rig (`sfm.rig: true`)**: the views of one frame share one centre with exactly
  known rotations, and COLMAP (3.12+) is told so (`rig_config.json`, `rig_configurator`,
  `--Mapper.ba_refine_sensor_from_rig 0`). It then solves one pose per 360 frame instead of
  one per view: faster mapping, no single view drifting off on its own, and each image is
  matched against every view of the neighbouring frames. A cube map **needs** the rig: its
  faces don't overlap, so without one COLMAP can't link them (a synthetic test registered
  50% without the rig vs 100% with it).
- `sfm.sequential_overlap_frames` now counts neighbouring **frames** (quadratic spacing:
  1, 2, 4, ... frames away). Old work folders with flat `f00012_v03.jpg` names still resume
  and run the old per-view way.

### Filling gaps from imperfect footage (real data only)

Three steps in the trainer (`scene_fill.py`, all on by default, all in `train.custom`):

- **Depth seeding** (`depth_seed`): where COLMAP found few points, the depth prior (already
  aligned to the SfM scale) is back-projected into extra starting Gaussians, so training has
  something to refine instead of an empty hole. Needs `depth_prior: true`.
- **Sky shell** (`sky_shell`): a sphere of Gaussians far outside the scene, coloured from the
  photos. Open sky and distant background draw there instead of rendering black or growing
  floaters. It's ordinary Gaussians in the `.ply`, so every viewer shows it.
- **Floater cleanup** (`prune`): before export, removes near-invisible Gaussians, specks
  hugging the flight path, isolated specks and oversized blobs. If that costs more than
  `prune_max_psnr_drop` held-out PSNR it is undone and a light pass (isolated specks and
  oversized blobs only) is tried instead. `ckpt.pt` keeps the uncleaned model; the
  held-out score is measured on the cleaned one. Trial scores are logged but kept out of
  `eval_history`.

`train_stats.json` → `fill` reports how many seeds, sky points and pruned Gaussians each
run had. To compare, run once with a step off (e.g. `"train": {"custom": {"prune": false}}`
in `CONFIG_OVERRIDES`) - the transform publishes it as the next `_vN`.

### Uneven light, haze and shimmer

- **Per-photo exposure correction** (`train.custom.exposure`, on): one small colour/brightness
  adjustment per photo soaks up the camera's auto exposure. Kept averaging to "no change", so
  the `.ply` holds the true colours. `train_stats.json` → `heldout_psnr_raw` scores the splat
  without any corrections (what viewers show); it should be close to `heldout_psnr`.
- **Bilateral grids** (`train.custom.bilagrid`, **off**): local per-photo colour grids.
  Tried and switched off: they let floaters near the camera hide behind per-photo
  corrections, which show up as dark, purple-ish blobs in viewers.
- **Anti-aliasing** (`train.custom.antialiased`, on): Mip-Splatting style filtering, so fine
  detail doesn't sparkle or thin out when you view from further away than the drone flew.
  Tick **Anti-aliased** in the viewer for these splats (untick it for older ones).
- **Gaussian budget** (`train.custom.cap_max: auto`): follows GPU memory: 3M on 16 GB (T4),
  4M on 24 GB (RTX 4090), 5M on 32 GB (RTX 5090), 6M on 48 GB+. Set a number to override.

### Popping, floaters and ghosting

Four things aimed at the flicker and blobs you see when you move around a splat:

- **Solid or gone** (`opacity_entropy`, from 60% of training until MCMC stops relocating at
  85%, `opacity_entropy_end`): pushes every Gaussian to be clearly opaque or clearly
  invisible. Thin see-through layers are what "pop" as a viewer re-sorts them. It stops when
  relocation stops because after that nothing recycles the Gaussians it pushes to ~0: on
  patio, running it to the end left 61% of the 3M Gaussians below 0.005 opacity, and
  removing them still cost 1.4 dB. `train_stats.json` → `opacity_shares` gives the
  near-invisible / see-through / solid split; `faint_share` is the see-through part (lower
  is better).
- **Camera refinement** (`pose_opt`): one tiny correction per 360 frame, shared by its six
  cube faces so the rig stays rigid. Fixes slight pose errors that show as doubled edges and
  ghosts. `train_stats.json` → `pose_refine` says how much the cameras moved (normally well
  under 1% of the scene and a fraction of a degree); the Fixer step reuses the same cameras.
- **Less angle-dependent colour** (`sh_degree: 2`): less sparkle from angles the drone
  never flew, and it matches what web viewers show.
- **Scene scaffold, built by Claude** (`scaffold`, needs the depth prior; see below).

`flythrough.mp4` is a short video along the flight path, shifted slightly sideways, so the
splat is seen from places no photo was taken. It shows popping and floaters that held-out
scores and still images miss: compare it between runs.

### Scene scaffold: Claude builds the scene's surfaces

Before training, the pipeline finds 20-40 candidate flat surfaces in the SfM points (level,
upright and sloped, split into connected pieces) and checks each one against every sampled
photo's depth estimate: how much of it the photos agree is there, how much they see
*through* it (glass), how much is hidden. A floor that the points show as several thin
layers on top of each other (slightly thicker or more tilted than the plane tolerance) is
joined back into one candidate first, so it isn't shared out between layers cell by cell. Photo agreement also fills in each surface's real
outline where a plain surface has few SfM points.

Then **Claude reviews the candidates** (when the pod has an Anthropic API key). It sees a
top-down map, photos along the flight and every candidate highlighted in its best photo, and
works with tools: more views, zoom into any photo, trim / extend / split an outline using a
region of a photo, merge pieces of the same surface, add surfaces the pipeline missed
(glass is fitted to its frame, since depth sees through glass), then give each a rule:

| rule | for | training |
|---|---|---|
| solid | paving, decking, walls, fences | exact depth where visible, nothing may float in front, splats lie flat |
| glass | windows, glass doors you mostly see reflections in | the same, placed from the outline; reflections become paint on the glass instead of blobs |
| window | glass you mostly see through (a lit room at dusk) | only the space in front must stay empty; the room behind keeps its own depth and parallax |
| soft | grass, gravel, low planting | only the space well in front must stay empty; never flattened |
| ignore | bushes, trees, furniture, anything uncertain | nothing |

It then checks its own work on a review sheet before finishing. Undecided candidates keep
the automatic rule (solid where the points or photos clearly say flat, else ignore), which is
also what's used without a key or if anything goes wrong; a run never fails because of it.

Outputs: `scaffold_review.jpg` (Claude's scaffold over photos and from above),
`scaffold_topdown.jpg`, `scaffold_candidates.jpg`, `scaffold_agent_log.md` (what it did and
why, with cost), `scaffold.json` (rules, stats and how often each surface's empty-space rule
fired during training) and `scaffold_decision.json` (the full decision: re-runs and
`--retrain` reuse it for free; edit a `rule` in it by hand to override Claude).

Settings (`train.custom`): `scaffold_mode` (claude | auto), `scaffold_model`
(claude-opus-5-5), `scaffold_max_cost` (USD cap per scene, default 3; the estimate uses
`scaffold_price_in/out` per million tokens), `scaffold_weight` (empty space; 0 = off),
`scaffold_flat_weight` (flat splats; 0 = off). Claude's instructions are in
`scaffold_prompt.md`; put your own version at `/workspace/scaffold_prompt.md` to change them
without rebuilding the image.

### AI fill-in with NVIDIA Fixer

What the real data can't show stays a hole or a smear. `fixer_refine.py` fills it in after
the fact, on a finished splat (`pod_run.py` does this automatically after each video):

1. re-cuts the exact training images from the video using the result's `scene/` bundle
   (poses, chosen frames, settings; published with every result),
2. measures how well the real photos cover every part of the splat (the coverage map, see
   below) and renders the splat from viewpoints the drone never had (off the path, turned
   between the cube faces), at Fixer's native 1024x576. Three in four of those viewpoints
   are picked because they look at gaps (`coverage_share`); the rest are random,
3. NVIDIA **Fixer** (Cosmos-Predict2 0.6B, single-step diffusion; the successor to Difix3D+)
   cleans each render,
4. checks each cleaned view: where real photos already pin the scene down, Fixer should
   barely change the render. Views it changed a lot there are trusted less, or dropped
   (`min_view_conf`). Then **Claude** looks at the views with the most made-up content and
   rejects any that invent objects or break the layout (`review: claude`, about $0.05 a
   round; `fixer_review.md` says what it rejected and why),
5. keeps training on the real photos plus the cleaned views (weighted lower, and by trust),
   learning each view mostly in its gaps: well covered pixels count only
   `observed_weight` (0.25), so Fixer can't repaint what the photos show but can still clear
   floaters there. Over a few rounds that move further from the path, it nudges every
   Gaussian to be clearly solid or gone (`opacity_entropy`), which clears see-through haze.

After each round it scores held-out real photos the way viewers see them (no per-photo
corrections); a round that lowers that by more than `max_round_drop` is undone and the
later (further-out) rounds are skipped. The best round is what's saved (`kept_round`;
0 = Fixer didn't help and the original splat is kept).

Output: `<name>_fixer/<name>_fixer.ply`, `coverage_topdown.jpg` (the gaps it aimed at), `fixer_stats.json` (scores before/after, per round)
and `fixer_preview/roundR_kk_render_fixer_photo.jpg` (render with its gaps tinted red |
Fixer | the real photo that shows the most of the same surfaces). Per round, `fixer_stats.json`
also says how many views were aimed at gaps, their mean gap share, the mean trust, and how
many were dropped or rejected. Settings: `fixer:` in `config.yaml` (`coverage: false` gives
the old behaviour: random views, every pixel learned equally).

### Coverage map: where the real photos don't reach

Every result gets `coverage_topdown.jpg` and `coverage.json` (`scene_coverage.py`). Each
Gaussian is scored 0-1 by how well the real photos pin it down. A photo counts when the
Gaussian is in its frame and not hidden behind something there, and counts less the
further away it was. Full coverage needs about 6 close-up sightings
(`coverage_good_views`) spread over at least 20 degrees (`coverage_good_spread`). One view,
or many from the same direction, can't fix where something is.

The map shows the scene from above: red = poorly covered, green = well covered, blue =
the flight path. The worst few gaps are numbered, and `coverage.json` lists them (centre,
mean coverage, distance from the flight path) with the share of the solid Gaussians that are
poorly covered. **Fly there next time**: a short extra pass beats any AI fill-in. The
Fixer step aims at the same gaps.

It runs in its own environment in the image (`/opt/fixer`, built by `pod/install_fixer.sh`;
`--build-arg WITH_FIXER=0` leaves it out), started by `fixer_fix.py`. It needs a bfloat16
GPU: RTX 30xx/40xx/50xx, A100, H100, L4/L40S - not the T4, so on Foundry's T4 only the main
pipeline runs. `selftest.py --with-fixer` (or `splat --selftest` on a pod) checks it.
If NVIDIA Transformer Engine (Fixer's fast kernels) is unavailable on a GPU, it switches to
an equivalent PyTorch path by itself and says so.

**Licence:** Fixer's weights are under the NVIDIA Open Model License (commercial use
allowed), its code under Apache-2.0. The refined splat contains AI-generated content; keep
the real-only one.

### Faster, metric SfM: the MapAnything + COLMAP hybrid

Set `sfm.mapper: mapanything` to replace COLMAP's slow incremental mapper:

1. **MapAnything** (Meta, Apache-2.0 weights) predicts **metric** camera poses for 4
   horizontal views per 360 frame, in overlapping windows, using our exact intrinsics.
   Each frame's views are fused into one rig pose (they share one physical camera), and
   the windows are stitched into one world.
2. **COLMAP** then triangulates points with those poses held fixed and bundle-adjusts
   (`refine_rounds` passes), so final precision comes from real feature matches.
3. Images COLMAP couldn't constrain (`min_observations`) are dropped, and the model is
   mapped back onto MapAnything's metric frame, so **the scene comes out in metres**.

If anything in this route fails, it logs why and falls back to `mapanything.fallback`
(default `colmap`), so a run never dies just because of it. `sfm_stats.json` records which
mapper actually ran, plus `colmap_vs_mapanything_rms_m` (how far COLMAP moved
MapAnything's cameras) and `view_agreement_deg` in `ma_init/report.json` (how consistent
MapAnything was within each frame).

**Install on Windows.** Use a separate venv: MapAnything pins its own OpenCV version, which
clashes with the main one. Git must be installed.

```
py -3.11 -m venv venv-ma
venv-ma\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install "mapanything @ git+https://github.com/facebookresearch/map-anything.git"
deactivate
```

then in `config.yaml`:

```yaml
sfm:
  mapper: mapanything
  mapanything:
    python: venv-ma/Scripts/python.exe
```

The first run downloads the model (a few GB). On an 8 GB GPU keep `window_frames` at 8-16;
if it runs out of memory, lower it. The RTX 2080 has no bf16, so it automatically uses fp16.
To compare against COLMAP alone, run the same video with both mappers and compare the
gate numbers, `--inspect` and held-out PSNR.

### Depth prior (steadier walls and floors from thin captures)

With `train.custom.depth_prior: true` (the default) the trainer first predicts a depth map for
every training image with a monocular model, then:

1. **aligns each map to the SfM points that image saw** (robust scale + shift, auto-detecting
   depth vs inverse depth); images whose prior doesn't fit the real geometry are skipped;
2. **only leans on it where geometry is weak**: areas full of tracked SfM points get a low
   weight (`depth_min_weight`), sparse areas get full weight (`depth_mode: selective`);
3. adds a disparity + depth-gradient loss that fades out over training.

Models: Depth Anything 3 mono if installed, otherwise Depth Anything V2 Small via
`transformers` (both Apache-2.0). The first run downloads ~100 MB and caches priors in
`work/<name>/depth_prior/`, so retraining reuses them.

Check the priors before a long training run (a few minutes, no training):

```
pip install transformers
python train_gsplat.py --depth-preview --data work/patio --config config.yaml
```

Look at `work\patio\depth_prior\preview\*.jpg`: photo | aligned depth | weight (white =
supervised). The log reports how many images got a prior and how many were rejected.
To compare fairly, retrain with `depth_prior: false` once and compare **held-out PSNR** and
the `eval\` images.

### Orientation and scale

- COLMAP doesn't know which way is up, so raw results can be tilted.
  With `train.custom.align: true` the trainer uses the known 360 view directions to
  level the scene, centre it on the flight path and face the drone's starting heading.
  `up_axis: "+y"` suits SuperSplat; use `"+z"` for Blender/Unreal.
- COLMAP units are arbitrary. Measure something real (e.g. the door) in the viewer and set
  `world_scale = real_size / viewer_size` in `config.yaml`, then retrain.

### Warping or blur? Run the inspector

```
python train_gsplat.py --inspect --data work/patio --config config.yaml
```

It checks that the views cut from each 360 frame agree on position and angle (they
physically share one camera). With the rig on they agree by construction, so the useful
lines become the up-direction agreement and the registered-views count. "Consistent" → geometry is fine; work on capture/training.
"INCONSISTENT" → SfM error (often glass/water reflections), which retraining won't fix.

---

### Self-test
(`--with-fixer` also loads and runs NVIDIA Fixer; `--only-fixer` checks just that.)

`python selftest.py` checks ffmpeg, COLMAP, torch/CUDA/gsplat, the depth model and MapAnything,
then runs the whole pipeline on a tiny synthetic 360 video (a few minutes on a GPU). Run it after
any setup change, and inside the Docker image before pushing it to Foundry
(`docker run --rm --gpus all <image> python /opt/app/selftest.py`).

## 4. Troubleshooting (everything we hit)

| Symptom | Cause | Fix |
|---|---|---|
| `zsh: command not found: pip` / `No module named 'cv2'` (Mac) | venv not active | `source .venv/bin/activate` |
| `Executable not found: ffprobe` / `'ffprobe' is not recognized` | PATH not refreshed after winget | Reopen terminal. Else find it: `where /R "%LOCALAPPDATA%\Microsoft\WinGet" ffprobe.exe` and add that `\bin` folder to *User Path* (Start → "environment variables"). Don't use `setx PATH`. |
| `... was unexpected at this time` from `COLMAP.bat` | .bat breaks on paths with spaces | Fixed in code (calls `bin\colmap.exe`). Also: no spaces in project path. |
| COLMAP exits instantly, code `3221226505` / `-1073740791`, no output | NVIDIA driver too old for the CUDA build (576.28 / CUDA 12.9 failed) | Update driver to 580+ and reboot. Temporary workaround: `sfm.use_gpu: false` |
| `SQLite error: unable to open database file` | Database path not writable (e.g. `C:\`) | Use a path inside the project |
| Mapper takes hours | Incremental mapper on ~1,800 images | `sfm.mapper: colmap_global` |
| SfM split into several models | Gaps/blurry stretches in the flight | Normal if small; the largest is used |
| `torch 2.6.0+cpu ... False` | Ran with `py` or outside the venv (system Python) | Activate venv, use `python` |
| `No suitable Python runtime found` for `py -3.12` | 3.12 not installed | Use `py -3.11` |
| `Permission denied ... .venv\Scripts\python.exe` / `Access is denied` deleting `.venv` | Old venv still in use (terminal, VS Code) | Close terminals + VS Code, `taskkill /f /im python.exe`, or reboot; or make a new venv named `venv` |
| A moved/renamed project's venv misbehaves | venvs hard-code their location | Delete and recreate it |
| `No module named 'packaging'` | Missing build dependency | `pip install packaging` |
| `gsplat: No CUDA toolkit found` then `'NoneType' ... 'CameraModelType'` | CUDA Toolkit missing or not visible | Install CUDA Toolkit 12.8, open a fresh x64 prompt. Or `set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8` and `set PATH=%CUDA_HOME%\bin;%PATH%` |
| `asm operand type size(8) does not match ... constraint 'r'` | 32-bit compiler (wrong prompt) | Use **x64** Native Tools prompt; `cl` must say `for x64` |
| `CUDACachingAllocator.h ... invalid combination of type specifiers` | torch 2.11 header clash on Windows | `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
| `cl : Command line error D8021 : invalid numeric argument '/Wno-attributes'` | gsplat Linux-only flag | Apply the patch in 2.9 |
| Build fails and you can't see why (only warnings visible) | Real error scrolled away | `python train_gsplat.py --check > build.log 2>&1` then `findstr /i /c:"error" build.log` |
| Any failed gsplat build | Stale half-built cache | `rmdir /s /q "%LOCALAPPDATA%\torch_extensions"` before retrying |
| GPU out of memory in training | Too many Gaussians / other apps | Lower `cap_max`, or `downscale: 2`, close GPU apps |
| Splat tilted / walls overhead | Arbitrary COLMAP axes | `align: true` (default) and retrain; or rotate in SuperSplat |
| Splat huge in viewer | Arbitrary COLMAP units | `world_scale`, or scale uniformly in SuperSplat |
| `No depth model could be loaded` | `transformers` missing, or no internet for the first download | `pip install transformers`, or set `depth_prior: false` |
| Depth prior step runs out of GPU memory | Batch too big | `depth_batch: 2` |
| Most priors rejected | Weak SfM fit for those images | Check `depth_prior\<tag>\*.skip` reasons; raise `depth_max_rel_error` cautiously |
| `MapAnything route failed ... Falling back` | MapAnything not installed in `mapanything.python`, out of GPU memory, or no internet on first run | Read the error above it; lower `window_frames`; check `mapanything.python` |
| Hybrid drops many images (`dropped_weak`) | Those views had too few feature matches to verify | Normal for sky/blank walls; if most are dropped, use `mapper: colmap` for that capture |

---

## 5. Mac notes

Homebrew `colmap` has no CUDA (`sfm.use_gpu: false`), and gsplat needs an NVIDIA GPU, so
on Apple Silicon run up to `--until-stage gate` and train with a Mac-capable trainer
(e.g. Brush) pointed at `work/<name>/`.

---

## 6. One entrypoint for everywhere

`foundry_app.py` detects where it's running and picks the right mode, so the same command
works on your PC, in Docker and in Foundry:

```
python foundry_app.py            # PC: processes raw\*.mp4 -> out\<name>\..., keeps work\
python foundry_app.py --detect   # just print what it detected
```

| Where | How it's detected | What it does |
|---|---|---|
| Foundry pipeline module | `BUILD2_TOKEN` + `RESOURCE_ALIAS_MAP` set | watches the input dataset, streams results |
| Foundry function module | `GET_JOB_URI` / `MODULE_AUTH_TOKEN` set | serves `start_splat` / `splat_status` |
| Docker on your PC | `/.dockerenv` or `/input` exists | `/input` -> `/output`, work in `/work` |
| Plain PC | none of the above | `raw\` -> `out\`, work kept in `work\` |

Videos that already have a `.ply` in the output are skipped (`SPLAT_FORCE=1` redoes them).
`SPLAT_MODE=pipeline|function|local` forces a mode. `splat_pipeline.py` still works directly
when you want stage-level control (`--from-stage`, `--until-stage`).

## 7. Running on Palantir Foundry

The project is packaged for Foundry compute modules: `Dockerfile`, `foundry_app.py`,
`requirements-container.txt`, `.dockerignore`. See **FOUNDRY.md** for the step-by-step
deployment guide (build, local test with `docker run --gpus all`, datasets, module
configuration, environment variables, troubleshooting).

## 8. Running on a rented GPU (RTX 5090 on RunPod / Vast.ai)

See **RENTING_A_GPU.md**: build the 5090 image once (`pod\build_5090.cmd`), start a pod,
upload videos, run `splat`, download the results. Same pipeline and settings as Foundry;
results are versioned the same way (`patio`, `patio_v2`, `patio_fixer`, ...).
