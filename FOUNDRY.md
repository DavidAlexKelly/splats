# Running the splat pipeline as a Foundry compute module

The same code that runs on your PC is packaged into one Docker image. In Foundry it reads
360 videos from an **input dataset** and writes splats into an **output dataset**:

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
| `foundry_app.py` | Entrypoint: `pipeline`, `function` or `local` mode |
| `requirements-container.txt` | Python packages for the image |
| `.dockerignore` | Keeps work folders and videos out of the image |
| `splat_pipeline.py`, `train_gsplat.py`, `config.yaml` | The pipeline itself (unchanged behaviour) |

---

## 0. Check with your Foundry admin first

- **Compute modules** must be enabled for your enrollment, and you need a **GPU** option
  (L4 or A10G with 24 GB is the sweet spot; T4 16 GB works with a smaller `cap_max`).
- GPU time costs compute-seconds. One 60 s patio clip is roughly 3 to 5 hours of a GPU replica.
- If this is a work enrollment, confirm it's an approved use before uploading personal footage.

---

## 1. Build the image

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

## 2. (Recommended) Test the exact image on your own GPU

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

## 3. Create the datasets

In a Foundry **Project** (the module and both datasets must be in the same Project):

1. Create a dataset, e.g. `360_videos`, and upload your DJI Studio export (`patio.mp4`) as a
   raw file (dataset → Upload → keep as files, don't parse as a table). 8K files are
   several GB, so the upload takes a while.
2. Create an empty dataset for results, e.g. `splats`.

## 4. Create the compute module

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
   - GPU: 1 × L4 or A10G (24 GB). T4: set `cap_max` to 2000000 (see below).
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

## 4b. Stream output (when pipeline mode only allows stream outputs)

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

## 5. Run it and watch

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

## 6. Settings you'll likely change

Everything in `config.yaml` can be overridden per deployment with `SPLAT_CONFIG_OVERRIDES`
(JSON, merged over the file), without rebuilding the image:

| Goal | Override |
|---|---|
| More detail on a 24–80 GB GPU | `{"train":{"custom":{"cap_max":5000000}}}` |
| Smaller GPU (T4) | `{"train":{"custom":{"cap_max":2000000}}}` |
| Real-world scale | `{"train":{"custom":{"world_scale":0.25}}}` |
| Faster SfM (try once stable) | `{"sfm":{"mapper":"colmap_global"}}` |
| Metric + faster SfM (MapAnything + COLMAP hybrid) | `{"sfm":{"mapper":"mapanything","mapanything":{"settings":{"window_frames":64}}}}` |
| Longer training | `{"train":{"iterations":40000}}` |

## 7. Alternative: function mode (trigger runs from Workshop / code)

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

## 8. Troubleshooting

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
