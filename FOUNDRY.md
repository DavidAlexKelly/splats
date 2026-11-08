# Running the splat pipeline on Palantir Foundry

There are two ways to run the same Docker image on Foundry:

| | **A. Lightweight transform (recommended)** | B. Compute module |
|---|---|---|
| What it is | A normal Foundry build that runs our container on a GPU | A long-running container service |
| Output | Real files straight into a dataset | Stream records + a rebuild transform (on this enrollment) |
| Billing | Only while the build runs | Whenever a replica is running, including idle time |
| Triggering | Build button, schedule, or "when input updates" | Polls the input every 5 minutes |
| Use it for | Turning videos into splats | An interactive service (e.g. a Workshop button) |

---

## A. Lightweight transform with your own container

```
360_videos dataset ──► Python transform (our image, T4 GPU) ──► splat_files dataset
  patio.mp4                                                       patio/patio.ply
  garden.mp4                                                      patio/patio.json, pipeline.log,
                                                                  train_stats.json, eval/*.jpg
                                                                  garden/...
```

The transform is **incremental**: each build processes only videos added since the last
successful build and appends their results. A video that fails gets `<name>/FAILED.txt`
(with the end of its log) instead of a `.ply`; if every video fails, the build fails.

### A1. Ask your admin for

- **Containerised workflows** enabled on the enrollment ("bring your own container").
- A **GPU resource queue** on the Project (only Projects with one can use GPUs).

### A2. Build and push the image as user 5001

BYOC requires the container user to be exactly 5001:

```
docker build --platform linux/amd64 --build-arg MAX_JOBS=4 --build-arg APP_UID=5001 -t splat-pipeline:0.2.1 .
set REPOSITORY=<your Artifacts repository rid>
set TOKEN=<fresh token>
echo %TOKEN%| docker login -u %REPOSITORY% --password-stdin <your container registry host>
docker tag splat-pipeline:0.2.1 <your container registry host>/splat:0.2.1
docker push <your container registry host>/splat:0.2.1
```

**Small changes later (minutes, not hours):** for code/config edits, or adding a missing
system tool, layer them on the image you already have instead of a full rebuild:

```
docker build --platform linux/amd64 -f Dockerfile.quick --build-arg BASE=splat-pipeline:0.2.1 -t splat-pipeline:0.2.2 .
```

Then tag/push `0.2.2` and set `IMAGE_TAG = "0.2.2"` in the transform. Use the full Dockerfile
only when dependencies change. The image must contain `curl` (Foundry's launcher uses it);
both Dockerfiles install it.

Foundry also requires each image layer to be under about 10 GB. If the push or the build is
rejected for layer size, build with `--build-arg WITH_MAPANYTHING=0`.

### A2b. Self-test the image on your PC before pushing (strongly recommended)

`selftest.py` checks every stage in a few minutes: tools, GPU + gsplat, the depth model,
MapAnything (if used), then a tiny end-to-end run of the real pipeline on a synthetic 360
video of a textured room. Run it on your own GPU before every push:

```
docker run --rm --gpus all splat-pipeline:0.2.1 python /opt/app/selftest.py
```

It ends with a summary and `ALL CHECKS PASSED`, or names the failing check and what to do.
`--quick` skips the end-to-end run. The transform also runs it on Foundry before touching any
video (`SELFTEST = True`, about 3-5 min on a T4); `SELFTEST_ONLY = True` runs just the test
with no videos and writes `_selftest/report.txt` to the output dataset.

### A3. Datasets (same Project as the repository)

- `360_videos`: upload the DJI exports as raw files (don't apply a schema).
- `splat_files`: leave it to be created by the first build, or create an empty one.

### A4. The Python Transforms repository

1. Create a **Pipelines → Python Transforms** repository in the same Project.
2. In the repository settings, add your **Artifacts repository** as a **backing
   repository**, so the transform can find the image.
3. Copy `foundry_transform/splat_transform.py` into `src/<your package>/` (next to the
   example transform the template created) and fill in the CONFIG block at the top:
   dataset paths, image name and tag, and optional `CONFIG_OVERRIDES`.
4. Commit, open the `splat_files` output and press **Build**. (Preview doesn't work for
   container transforms; only full builds do.)

The build log shows the whole pipeline live, prefixed with `| `: the GPU check, each
stage, training progress and the uploads.

If you get "`function object has no attribute using`", the repository's transforms
library is too old: upgrade the repository (the docs' repository upgrade guide), or ask
me for the legacy `@lightweight` syntax.

### A5. Settings

Edit `CONFIG_OVERRIDES` in the transform, for example:

```python
CONFIG_OVERRIDES = {"sfm": {"mapper": "mapanything",
                            "mapanything": {"settings": {"window_frames": 24}}}}
```

Resources are set at the top too (`CPU_CORES`, `MEMORY_GB`, `GPU_TYPE`). The defaults
(8 cores, 32 GB, `NVIDIA_T4`) suit a T4. 8 cores and 64 GB are the default upper limits.

**Unknowns to check on the first run:** local disk space inside the build container (an
8K video needs tens of GB of temporary space) and any maximum build duration. Start with a
10-15 second clip.

### A6. Difix fill-in transform (optional)

`foundry_transform/difix_transform.py` is a second transform in the same repository, same
image, same T4. It refines splats the main transform already made; it doesn't touch them.

1. Create an empty dataset (e.g. `splat_files_difix`) and put its RID in `DIFIX_DATASET`.
2. Results to refine need a `scene/` folder. Results from before that update don't have one:
   bump `RERUN` in `splat_transform.py` once to make a new version of them.
3. `TARGETS = ["latest"]` refines the newest version of every video; or name folders,
   e.g. `["patio_v4"]`. Build the output dataset.

Takes about 30-60 min per scene on a T4 (3 rounds x 200 views). `DIFIX_OVERRIDES =
{"difix": {"rounds": 2, "views_per_round": 120}}` is a quicker first look. Each target gets
`<folder>_difix/` (`_difix_v2`, ... on re-runs); a failed one gets `FAILED.txt` and is tried
again next build. Licence: non-commercial research/evaluation only (see README).

The image needs the Difix environment: the full `Dockerfile` and `Dockerfile.quick` both add it
by default (about 5 GB more, downloaded once at build time).

---

## B. Compute module

The same image also runs as a compute module. In Foundry it reads 360 videos from an
**input dataset** and writes splats out:

```
input dataset "videos"                 output dataset "splats"
  patio.mp4            ──►  GPU  ──►     patio/patio.ply          ← the splat
  garden.mp4              module          patio/patio.json         ← run summary
                                          patio/train_stats.json   ← held-out PSNR etc.
                                          patio/pipeline.log       ← full log
                                          patio/eval/*.jpg         ← photo | splat comparisons
                                          garden/...
```

Videos already done (their `.ply` exists in the output) are skipped; failures write
`<name>/FAILED.json` and aren't retried unless `SPLAT_RETRY_FAILED=1`. New videos dropped
into the input dataset are picked up automatically (polled every 5 minutes).

Files in the image:

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the image: CUDA 12.6, ffmpeg, COLMAP (CUDA), PyTorch 2.8, gsplat (prebuilt) |
| `Dockerfile.quick` | Fast rebuild: layers code/config changes and tool fixes on an existing image |
| `foundry_app.py` | Entrypoint: `pipeline`, `function` or `local` mode |
| `requirements-container.txt` | Python packages for the image |
| `.dockerignore` | Keeps work folders and videos out of the image |
| `splat_pipeline.py`, `train_gsplat.py`, `config.yaml` | The pipeline itself (unchanged behaviour) |

---

### B0. Check with your Foundry admin first

- **Compute modules** must be enabled for your enrollment, and you need a **GPU** option
  (a T4 with 16 GB is enough: it's the same GPU generation as the RTX 2080 the pipeline was
  tested on, with twice the memory; L4/A10G are faster).
- GPU time costs compute-seconds. One 60 s patio clip is roughly 3 to 5 hours of a GPU replica.
- If this is a work enrollment, confirm it's an approved use before uploading personal footage.

---

### B1. Build the image

You need Docker (on Windows: **Docker Desktop** with the WSL 2 backend). From the `splats` folder:

```
docker build --platform linux/amd64 --build-arg MAX_JOBS=4 -t splat-pipeline:0.1.0 .
```

The first build takes a while, roughly 30–60 minutes on an x86 PC, because COLMAP and
gsplat's CUDA kernels are compiled.

**Build on the Windows PC if you can.** On an Apple Silicon Mac the amd64 image is built
under emulation: much slower (hours), more memory-hungry, and the Mac can't run the GPU
test in step 2. If you must build on the Mac, give Docker Desktop at least 12 GB of memory
(Settings → Resources) and use `--build-arg MAX_JOBS=2`.

**If the build dies with `Killed signal terminated program cc1plus` or `cannot allocate
memory`**, it ran out of RAM compiling COLMAP: lower `MAX_JOBS` (e.g. `--build-arg
MAX_JOBS=2`) and/or give Docker more memory, then rerun. Finished steps are cached. Rebuilds after code-only changes are quick (cached layers).
The build ends by printing `gsplat 1.5.3 prebuilt kernels: True`.

**Never tag it `latest`**: Foundry requires a specific tag. Bump the tag (0.1.1, 0.1.2...)
each time you push a change.

### B2. (Recommended) Test the exact image on your own GPU

This catches problems before you spend Foundry GPU time. Docker Desktop's GPU support needs a
current NVIDIA driver (you have one). From the `splats` folder:

```
mkdir test_out
docker run --rm --gpus all -e SPLAT_MODE=local ^
  -v "%cd%\raw:/input" -v "%cd%\test_out:/output" splat-pipeline:0.1.0
```

Look for `gpu check: OK: rendered and back-propagated...` in the first lines. For a quick
smoke test, put a short clip (10–15 s) in a separate folder and mount that as `/input`.
Results appear in `test_out\<name>\`.

### B3. Create the datasets

In a Foundry **Project** (the module and both datasets must be in the same Project):

1. Create a dataset, e.g. `360_videos`, and upload your DJI Studio export (`patio.mp4`) as a
   raw file (dataset → Upload → keep as files, don't parse as a table). 8K files are
   several GB, so the upload takes a while.
2. Create an empty dataset for results, e.g. `splats`.

### B4. Create the compute module

1. In the Project: **+ New → Compute Module**, choose a **pipeline** module.
2. Follow the module's container/image instructions to push your image to its Artifacts
   repository. Foundry shows the exact `docker login`, `docker tag` and `docker push`
   commands for your enrollment; use tag `0.1.0`, not `latest`.
3. **Execution mode:** Pipeline.
   - Add **input**: the `360_videos` dataset with identifier **`videos`**.
   - Add **output**: the `splats` dataset with identifier **`splats`**.
4. **Source / network egress:** add a source whose network policy allows your Foundry
   hostname (e.g. `yourenrollment.palantirfoundry.com`). Foundry's docs require this for
   the container to call Foundry's APIs, even in pipeline mode.
5. **Resources** (per replica):
   - GPU: 1 × T4 (16 GB) works with the defaults; L4/A10G (24 GB) are faster.
   - CPU: 8 or more (COLMAP's mapper is CPU-heavy).
   - Memory: 32 GB or more.
   - Storage: mount a volume at **`/work`** with at least **60 GB**. Each 8K video needs
     tens of GB of temporary frames and images. Without it, the container's ephemeral disk
     may fill up.
6. **Scaling:** minimum replicas **1** (pipeline modules receive no queries, so otherwise
   the autoscaler can stop it), maximum 1.
7. **Environment variables:**

   | Variable | Value |
   |---|---|
   | `SPLAT_FOUNDRY_URL` | your hostname, e.g. `yourenrollment.palantirfoundry.com` |
   | `SPLAT_CONFIG_OVERRIDES` | optional JSON, e.g. `{"train":{"custom":{"cap_max":4000000}}}` |
   | `SPLAT_POLL_SECONDS` | optional, default `300` |
| `SPLAT_OUTPUT_KIND` | `stream` (default) or `dataset` |
| `SPLAT_MODE` | `auto` (default: detects pipeline vs function mode) |
| `SPLAT_STREAM_CHUNK_KB` | optional, default `512` |

### B4b. Stream output (when pipeline mode only allows stream outputs)

This is the default (`SPLAT_OUTPUT_KIND=stream`). Each result file is split into chunks and
sent as text records; a Python transform then rebuilds the real files into a dataset.

```
videos dataset ──► compute module ──► splat_stream (records) ──► transform ──► splat_files dataset
                                       patio/patio.ply chunks 0..N            patio/patio.ply
                                       patio/eval/... chunks                   patio/eval/...
                                       run summary                             manifest.json
```

**1. Create the output stream** (in the module's Project), named e.g. `splat_stream`. The
ready-to-paste JSON schema is in `foundry_transform/splat_stream_schema.json`; the fields are:

| Field | Type |
|---|---|
| `record_type` | String |
| `run_id` | String |
| `video` | String |
| `file_path` | String |
| `chunk_index` | Integer |
| `chunk_count` | Integer |
| `total_bytes` | Long |
| `sha256` | String |
| `data_b64` | String |
| `info` | String |
| `created_at` | String |

**2. Module configuration:** input dataset `videos` and output stream `splats` (pointing at
`splat_stream`). Keep the `/work` volume: the module remembers which videos it has already
published in `/work/_state/published.json` (a stream can't be listed back reliably), so
if `/work` isn't persistent, a restart re-processes videos.

**3. Record size.** Each record carries `SPLAT_STREAM_CHUNK_KB` of file data (default 512 KB,
about 700 KB of text). If the logs say the stream rejected a record as too large, set
`SPLAT_STREAM_CHUNK_KB=256` (or 128). A 750 MB splat at 512 KB is about 1,500 records;
the 8K test clip is far fewer.

**4. Rebuild transform.** Create a **Python transforms** Code Repository in the same
Project, add `foundry_transform/reconstruct_splats.py` from this project (e.g. under
`src/<your package>/datasets/`), set `STREAM_PATH` and `OUTPUT_PATH` at the top, commit,
and build. Schedule it to run when `splat_stream` updates (or hourly).

The output dataset then holds the real files plus `manifest.json` (`verified: true` means
the file matched its checksum). Chunks may arrive in any order or twice. Files are only
written once every chunk is present and the checksum matches, and if a video is processed
again, the newest complete run wins.

Batch transforms read the stream's *archived* records, so a just-finished run can take a
little while to show up. If the transform finds no or partial files, rebuild again shortly.

**Fallback without the transform:** export the stream's records (Parquet/CSV/JSON lines)
and run locally:

```
python splat_stream.py rebuild exported_records.parquet rebuilt_files
```

### B5. Run it and watch

Start the module and open its **logs**. A healthy run looks like:

```
splat compute module starting in pipeline mode
gpu check: GPU: NVIDIA L4
gpu check: OK: rendered and back-propagated through gsplat on the GPU
1 videos in input, 1 to process
Downloading patio.mp4
[patio] === extract === ... === sfm === ... === train ===
step  5000/30000  loss ...  psnr ...  gaussians ...
EVAL step 5000: held-out PSNR ...
uploading patio/patio.ply (750.3 MB)
patio: ok, published 12 files
```

Then open the `splats` dataset, download `patio/patio.ply` and drag it into SuperSplat.
Look at `patio/eval/*.jpg` to see photo-vs-splat comparisons.

### B6. Settings you'll likely change

Everything in `config.yaml` can be overridden per deployment with `SPLAT_CONFIG_OVERRIDES`
(JSON, merged over the file), without rebuilding the image:

| Goal | Override |
|---|---|
| More detail on a 24–80 GB GPU | `{"train":{"custom":{"cap_max":5000000}}}` |
| T4, hybrid mapper | `{"sfm":{"mapper":"mapanything","mapanything":{"settings":{"window_frames":24}}}}` |
| Real-world scale | `{"train":{"custom":{"world_scale":0.25}}}` |
| Faster SfM (try once stable) | `{"sfm":{"mapper":"colmap_global"}}` |
| Metric + faster SfM (MapAnything + COLMAP hybrid) | `{"sfm":{"mapper":"mapanything","mapanything":{"settings":{"window_frames":64}}}}` |
| Longer training | `{"train":{"iterations":40000}}` |

### B7. Alternative: function mode (trigger runs from Workshop / code)

Choose a **Functions** module (the image detects function mode automatically; `SPLAT_MODE=function` forces it) with **Application
permissions** (not available on every enrollment). Give the module's service user read
access to the video dataset and write access to the output dataset. Also add the egress
source and set `SPLAT_FOUNDRY_URL`. Functions exposed:

- `start_splat(input_dataset_rid, video_path, output_dataset_rid, branch, config_overrides_json)`
  queues a job and returns its `job_id` straight away (`branch` usually `master`,
  `config_overrides_json` can be empty).
- `splat_status(job_id)` returns status, current stage, last log line and output path.
- `list_splat_jobs(job_id)` returns all jobs as JSON (`job_id` is ignored).

Results are written in a single transaction per video.

**Scaling: set minimum AND maximum replicas to 1.** Foundry's architecture docs say you
can't rely on state inside a replica, and replicas can't talk to each other. Job status
lives in memory, so with 2+ replicas `splat_status` can hit a replica that never saw the
job; with 0 the module can scale down mid-job. A restart loses the job list (not the
results, which are only ever written as a complete, committed transaction).

**Calling the functions:** after starting the module from its **Overview** page, use the
**Query** panel at the bottom to call `start_splat` / `splat_status`. To use them from
Workshop or Slate, open the **Functions** tab, choose **Import**, and select them.

**If pipeline mode only offers stream outputs:** Foundry's docs say pipeline outputs can
be datasets, streams or media sets, so first check the output picker for a resource-type
option and that the datasets are in the *same Project* as the module. If datasets still
aren't offered, use function mode.

### B8. Troubleshooting

| Symptom in the logs | Fix |
|---|---|
| `GPU self-check FAILED` | No GPU on the replica, or the host driver is too old for CUDA 12.6 |
| `Foundry hostname unknown` | Set `SPLAT_FOUNDRY_URL` |
| `No resource with identifier 'videos'` | Input/output identifiers must be exactly `videos` / `splats` (or set `SPLAT_INPUT_ALIAS` / `SPLAT_OUTPUT_ALIAS`) |
| Connection errors / timeouts calling Foundry | Missing egress source for your Foundry hostname |
| `list files failed: HTTP 401/403` | Dataset not attached as input/output, or not in the same Project |
| `upload ... failed: HTTP 409` (pipeline mode) | Output dataset has a conflicting open transaction. Retry; if persistent, use function mode (explicit transactions) and tell me the error text |
| Out of disk / `No space left` | Mount a bigger volume at `/work` |
| GPU out of memory | Lower `cap_max` via `SPLAT_CONFIG_OVERRIDES` |
| Module stopped on its own | Minimum replicas must be 1 |

**What's verified and what isn't.** The dataset logic (listing, skipping finished videos,
downloading, uploading with and without transactions, failure handling) was tested
against a mock of Foundry's dataset API. The real platform hasn't been exercised. The
parts most likely to need a small tweak on first deploy are the exact upload behaviour in
pipeline mode and your enrollment's network and volume settings. The pipeline log will
show the precise HTTP error if so.
