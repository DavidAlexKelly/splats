#!/usr/bin/env python3
"""
foundry_app.py - runs the 360 video -> Gaussian splat pipeline as a Palantir Foundry
compute module (or locally in the same container for testing).

Run it the same way everywhere:  python foundry_app.py
It detects where it is (SPLAT_MODE=auto, the default) and picks the mode itself:

  Foundry, pipeline mode  BUILD2_TOKEN + RESOURCE_ALIAS_MAP are set   -> pipeline
  Foundry, function mode  GET_JOB_URI / MODULE_AUTH_TOKEN are set     -> function
  Docker on a PC          /.dockerenv or an /input folder exists     -> local (/input, /output, /work)
  Plain PC (no container) none of the above                          -> local (./raw, ./out, ./work)

`python foundry_app.py --detect` prints what it found and exits. Set SPLAT_MODE to force a mode:

  pipeline (default)  Foundry pipeline mode. Attach an input dataset of videos (alias
                      SPLAT_INPUT_ALIAS, default "videos") and an output (alias
                      SPLAT_OUTPUT_ALIAS, default "splats"). The output is a STREAM by
                      default (SPLAT_OUTPUT_KIND=stream): result files are sent as base64
                      chunk records (see splat_stream.py) and rebuilt into real files by
                      the reconstruct transform. SPLAT_OUTPUT_KIND=dataset writes files
                      directly where the enrollment allows dataset outputs. New videos are
                      picked up by polling every SPLAT_POLL_SECONDS.

  function            Foundry function mode with application permissions. Exposes
                      start_splat / splat_status / list_splat_jobs functions. Jobs run one
                      at a time on a background worker.

  local               No Foundry. Processes videos in SPLAT_LOCAL_INPUT (default /input)
                      and writes results to SPLAT_LOCAL_OUTPUT (default /output). Use with
                      `docker run --gpus all` to test the image before uploading it.

Other environment variables:
  SPLAT_CONFIG          base config file (default: config.yaml next to this file)
  SPLAT_CONFIG_OVERRIDES JSON merged over the config, e.g. {"train":{"custom":{"cap_max":4000000}}}
  SPLAT_WORK            working directory (default /work in containers, ./work on a PC)
  SPLAT_KEEP_WORK       "1" keeps per-video work folders (default: keep on a PC, delete in a container)
  COLMAP_BIN            colmap executable; overrides sfm.colmap_bin only when set
  SPLAT_FOUNDRY_URL     Foundry hostname if FOUNDRY_URL isn't provided by the platform
  SPLAT_RETRY_FAILED    "1" retries videos that previously failed (pipeline mode)
  SPLAT_STREAM_CHUNK_KB raw KB per stream record (default 512 -> ~700 KB of base64 text)
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import yaml

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import splat_pipeline as sp  # noqa: E402
import splat_stream as ss  # noqa: E402

VIDEO_EXTS = sp.VIDEO_EXTS


# --------------------------------------------------------------------------- #
# Logging (stdout is collected by the compute module)
# --------------------------------------------------------------------------- #
def get_log(name: str) -> logging.Logger:
    try:
        from compute_modules.logging import get_logger
        lg = get_logger(name)
    except Exception:
        lg = logging.getLogger(name)
        if not lg.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
            lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg


log = get_log("splat_app")


def setup_pipeline_logging():
    """Send the pipeline's own log (stages, COLMAP summary, training progress) to stdout."""
    plog = sp.log
    plog.setLevel(logging.DEBUG)
    if not any(getattr(h, "_splat_console", False) for h in plog.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        h._splat_console = True
        plog.addHandler(h)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def effective_config(overrides: dict | None = None) -> Path:
    """Base config + container defaults + env/JSON overrides, written to a real file so the
    trainer subprocess reads exactly the same settings. Returns the file path."""
    base = Path(os.environ.get("SPLAT_CONFIG", APP_DIR / "config.yaml"))
    cfg = yaml.safe_load(base.read_text()) if base.exists() else {}
    if os.environ.get("COLMAP_BIN"):  # set in the Docker image; on a PC config.yaml decides
        cfg = sp.deep_merge(cfg, {"sfm": {"colmap_bin": os.environ["COLMAP_BIN"]}})
    env_over = os.environ.get("SPLAT_CONFIG_OVERRIDES", "").strip()
    if env_over:
        cfg = sp.deep_merge(cfg, json.loads(env_over))
    if overrides:
        cfg = sp.deep_merge(cfg, overrides)
    out = Path(tempfile.gettempdir()) / f"splat_config_{uuid.uuid4().hex[:8]}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def work_root() -> Path:
    p = Path(os.environ.get("SPLAT_WORK") or ENV.get("work", "/work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def gpu_self_check():
    r = subprocess.run([sys.executable, str(APP_DIR / "train_gsplat.py"), "--check"],
                       capture_output=True, text=True)
    for line in (r.stdout + r.stderr).strip().splitlines()[-6:]:
        log.info("gpu check: %s", line)
    if r.returncode != 0:
        log.error("GPU self-check FAILED. Is a GPU attached to this compute module replica?")
    return r.returncode == 0


def run_one(video: Path, cfg_path: Path, out_dir: Path) -> dict:
    """Run the full pipeline for one local video file; returns the run_pipeline result."""
    setup_pipeline_logging()
    cfg = sp.load_config(cfg_path)
    res = sp.run_pipeline([video], cfg, work_root(), out_dir)
    return res[video.name]


def result_files(video_stem: str, res: dict, out_dir: Path) -> list[tuple[Path, str]]:
    """(local file, path inside the output dataset) pairs to publish for one video."""
    files = []
    for f in (out_dir / f"{video_stem}.ply", out_dir / f"{video_stem}.json"):
        if f.exists():
            files.append((f, f"{video_stem}/{f.name}"))
    wd = Path(res.get("work_dir") or "")
    extra = [wd / "pipeline.log", wd / "sfm_stats.json", wd / "train" / "train_stats.json",
             wd / "train" / "alignment.json"]
    for f in extra:
        if f.exists():
            files.append((f, f"{video_stem}/{f.name}"))
    for f in sorted((wd / "train" / "eval").glob("*.jpg")) if wd.exists() else []:
        files.append((f, f"{video_stem}/eval/{f.name}"))
    scene = out_dir / f"{video_stem}_scene"  # small bundle for later refinement (difix_refine.py)
    for f in sorted(scene.rglob("*")) if scene.exists() else []:
        if f.is_file():
            files.append((f, f"{video_stem}/scene/{f.relative_to(scene).as_posix()}"))
    return files


def fresh_out_dir(stem: str) -> Path:
    """A clean per-video output folder, so a failed run can never re-publish stale results."""
    d = work_root() / "_out" / stem
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def cleanup(res: dict):
    keep = os.environ.get("SPLAT_KEEP_WORK")
    if keep == "1" or (keep is None and ENV.get("where") == "pc"):
        return
    wd = res.get("work_dir")
    if wd and Path(wd).exists():
        shutil.rmtree(wd, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Foundry dataset API (public v2 API, with the documented catalog endpoints as fallback)
# --------------------------------------------------------------------------- #
class FoundryDatasets:
    def __init__(self, hostname: str, token_fn):
        import requests
        self.requests = requests
        self.host = hostname.replace("https://", "").replace("http://", "").strip("/")
        self.base = f"https://{self.host}"
        self.token_fn = token_fn
        ca = os.environ.get("DEFAULT_CA_PATH")
        self.verify = ca if ca and Path(ca).exists() else True

    def _h(self, extra=None):
        h = {"Authorization": f"Bearer {self.token_fn()}"}
        h.update(extra or {})
        return h

    def _check(self, r, what):
        if r.status_code >= 400:
            raise RuntimeError(f"{what} failed: HTTP {r.status_code}: {r.text[:500]}")
        return r

    def list_files(self, rid: str, branch: str) -> list[str]:
        paths, token = [], None
        url = f"{self.base}/api/v2/datasets/{rid}/files"
        try:
            while True:
                params = {"branchName": branch, "pageSize": 1000}
                if token:
                    params["pageToken"] = token
                r = self._check(self.requests.get(url, params=params, headers=self._h(),
                                                  verify=self.verify, timeout=120), "list files")
                body = r.json()
                paths += [f["path"] for f in body.get("data", [])]
                token = body.get("nextPageToken")
                if not token:
                    return paths
        except RuntimeError as e:
            log.warning("v2 file listing failed (%s); trying catalog endpoint", e)
        url = f"{self.base}/foundry-catalog/api/catalog/datasets/{rid}/views2/{branch}/files"
        paths, token = [], None
        while True:
            params = {"pageSize": 1000, "includeOpenExclusiveTransaction": "false",
                      "excludeHiddenFiles": "true"}
            if token:
                params["pageStartLogicalPath"] = token
            r = self._check(self.requests.get(url, params=params, headers=self._h(),
                                              verify=self.verify, timeout=120), "catalog list")
            body = r.json()
            paths += [f["logicalPath"] for f in body.get("values", [])]
            token = body.get("nextPageToken")
            if not token:
                return paths

    def download(self, rid: str, path: str, dest: Path, branch: str):
        dest.parent.mkdir(parents=True, exist_ok=True)
        urls = [(f"{self.base}/api/v2/datasets/{rid}/files/{quote(path, safe='')}/content",
                 {"branchName": branch}),
                (f"{self.base}/foundry-data-proxy/api/dataproxy/datasets/{rid}/views/{branch}/"
                 f"{quote(path)}", {})]
        last = None
        for url, params in urls:
            with self.requests.get(url, params=params, headers=self._h(), stream=True,
                                   verify=self.verify, timeout=600) as r:
                if r.status_code >= 400:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    continue
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return
        raise RuntimeError(f"download {path} failed: {last}")

    def upload(self, rid: str, local: Path, path: str, branch: str | None = None,
               transaction_rid: str | None = None):
        url = f"{self.base}/api/v2/datasets/{rid}/files/{quote(path, safe='')}/upload"
        params = {"transactionRid": transaction_rid} if transaction_rid else {"branchName": branch}
        with open(local, "rb") as f:
            r = self.requests.post(url, params=params, data=f, verify=self.verify, timeout=3600,
                                   headers=self._h({"Content-Type": "application/octet-stream"}))
        self._check(r, f"upload {path}")

    def create_transaction(self, rid: str, branch: str, kind: str = "APPEND") -> str:
        r = self.requests.post(f"{self.base}/api/v2/datasets/{rid}/transactions",
                               params={"branchName": branch}, json={"transactionType": kind},
                               headers=self._h(), verify=self.verify, timeout=120)
        return self._check(r, "create transaction").json()["rid"]

    def commit(self, rid: str, tx: str):
        r = self.requests.post(f"{self.base}/api/v2/datasets/{rid}/transactions/{tx}/commit",
                               headers=self._h(), verify=self.verify, timeout=300)
        self._check(r, "commit transaction")

    def abort(self, rid: str, tx: str):
        try:
            self.requests.post(f"{self.base}/api/v2/datasets/{rid}/transactions/{tx}/abort",
                               headers=self._h(), verify=self.verify, timeout=120)
        except Exception:
            pass

    def publish(self, rid: str, branch: str, files: list[tuple[Path, str]], use_transaction: bool):
        """Upload files. With use_transaction, everything lands atomically in one APPEND
        transaction; otherwise each upload goes to the branch directly."""
        if not use_transaction:
            for local, path in files:
                log.info("uploading %s (%.1f MB)", path, local.stat().st_size / 1e6)
                self.upload(rid, local, path, branch=branch)
            return
        tx = self.create_transaction(rid, branch)
        try:
            for local, path in files:
                log.info("uploading %s (%.1f MB)", path, local.stat().st_size / 1e6)
                self.upload(rid, local, path, transaction_rid=tx)
            self.commit(rid, tx)
        except Exception:
            self.abort(rid, tx)
            raise


class StreamWriter:
    """Writes JSON records to a Foundry stream via stream-proxy (the endpoint used in
    Foundry's pipeline-mode example), with retries."""

    def __init__(self, api: FoundryDatasets, rid: str, branch: str):
        self.api, self.rid, self.branch = api, rid, branch
        self.url = (f"{api.base}/stream-proxy/api/streams/{rid}/branches/{branch}/jsonRecord")

    def put(self, record: dict):
        delay = 2.0
        for attempt in range(6):
            try:
                r = self.api.requests.post(self.url, json=record, headers=self.api._h(),
                                           verify=self.api.verify, timeout=120)
                if r.status_code < 400:
                    return
                if r.status_code == 413 or "too large" in r.text.lower():
                    raise RuntimeError("Stream rejected the record as too large: lower "
                                       "SPLAT_STREAM_CHUNK_KB (e.g. 256)")
                if r.status_code in (400, 401, 403, 404):
                    raise RuntimeError(f"stream write failed: HTTP {r.status_code}: {r.text[:400]}")
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except self.api.requests.RequestException as e:
                err = str(e)
            log.warning("stream write retry %d (%s)", attempt + 1, err)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"stream write failed after retries: {err}")

    def publish(self, files: list[tuple[Path, str]], video: str, summary: dict):
        run_id = uuid.uuid4().hex
        chunk = int(os.environ.get("SPLAT_STREAM_CHUNK_KB", "512")) * 1024
        for local, path in files:
            n = 0
            for rec in ss.file_records(local, path, run_id, video, chunk):
                self.put(rec)
                n += 1
            log.info("streamed %s (%.1f MB, %d records)", path, local.stat().st_size / 1e6, n)
        self.put(ss.run_record(run_id, video, summary))


class LocalState:
    """Remembers which input videos were already published (stream outputs can't be listed
    back reliably). Lives in the /work volume, so it survives restarts if /work is persistent."""

    def __init__(self):
        self.path = work_root() / "_state" / "published.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, status):
        self.data[key] = {"status": status, "at": ss.now_iso()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def foundry_hostname() -> str:
    h = os.environ.get("FOUNDRY_URL") or os.environ.get("SPLAT_FOUNDRY_URL")
    if not h:
        raise RuntimeError("Foundry hostname unknown: set SPLAT_FOUNDRY_URL "
                           "(e.g. yourenrollment.palantirfoundry.com) in the module config")
    return h


# --------------------------------------------------------------------------- #
# Pipeline mode
# --------------------------------------------------------------------------- #
def pipeline_mode():
    token_file = os.environ.get("BUILD2_TOKEN")
    alias_file = os.environ.get("RESOURCE_ALIAS_MAP")
    if not token_file or not alias_file:
        raise RuntimeError("BUILD2_TOKEN / RESOURCE_ALIAS_MAP not set: is the module in "
                           "pipeline mode with inputs and outputs attached?")

    def token():
        return Path(token_file).read_text().strip()  # re-read: the platform may rotate it

    aliases = json.loads(Path(alias_file).read_text())
    in_alias = os.environ.get("SPLAT_INPUT_ALIAS", "videos")
    out_alias = os.environ.get("SPLAT_OUTPUT_ALIAS", "splats")
    for a in (in_alias, out_alias):
        if a not in aliases:
            raise RuntimeError(f"No resource with identifier '{a}'. Configured: {list(aliases)}")
    in_rid, in_branch = aliases[in_alias]["rid"], aliases[in_alias].get("branch") or "master"
    out_rid, out_branch = aliases[out_alias]["rid"], aliases[out_alias].get("branch") or "master"
    api = FoundryDatasets(foundry_hostname(), token)
    poll = int(os.environ.get("SPLAT_POLL_SECONDS", "300"))
    retry_failed = os.environ.get("SPLAT_RETRY_FAILED") == "1"
    kind = os.environ.get("SPLAT_OUTPUT_KIND", "stream").lower()
    log.info("Pipeline mode: input %s (%s), %s output %s (%s)", in_rid, in_branch, kind,
             out_rid, out_branch)
    gpu_self_check()
    if kind == "stream":
        stream = StreamWriter(api, out_rid, out_branch)
        state = LocalState()
        while True:
            try:
                videos = [p for p in api.list_files(in_rid, in_branch)
                          if Path(p).suffix.lower() in VIDEO_EXTS]
                todo = [p for p in videos if not (st := state.get(f"{in_rid}:{p}")) or
                        (st["status"] != "ok" and retry_failed)]
                log.info("%d videos in input, %d to process", len(videos), len(todo))
                for p in todo:
                    res = process_remote_video(api, in_rid, in_branch, p, None, None, None,
                                               False, stream=stream)
                    state.set(f"{in_rid}:{p}", res["status"])
            except Exception as e:
                log.error("Polling cycle failed: %s\n%s", e, traceback.format_exc())
            time.sleep(poll)

    while True:
        try:
            videos = [p for p in api.list_files(in_rid, in_branch)
                      if Path(p).suffix.lower() in VIDEO_EXTS]
            done = set(api.list_files(out_rid, out_branch))
            todo = []
            for p in videos:
                stem = Path(p).stem
                if f"{stem}/{stem}.ply" in done:
                    continue
                if f"{stem}/FAILED.json" in done and not retry_failed:
                    continue
                todo.append(p)
            log.info("%d videos in input, %d to process", len(videos), len(todo))
            for p in todo:
                process_remote_video(api, in_rid, in_branch, p, out_rid, out_branch,
                                     overrides=None, use_transaction=False)
        except Exception as e:
            log.error("Polling cycle failed: %s\n%s", e, traceback.format_exc())
        time.sleep(poll)


def process_remote_video(api: FoundryDatasets, in_rid, in_branch, path, out_rid, out_branch,
                         overrides, use_transaction, status_cb=None,
                         stream: "StreamWriter | None" = None) -> dict:
    stem = Path(path).stem
    job_dir = work_root() / "_downloads"
    local_video = job_dir / Path(path).name
    status = status_cb or (lambda **kw: None)
    try:
        status(stage="downloading")
        log.info("Downloading %s", path)
        api.download(in_rid, path, local_video, in_branch)
        status(stage="processing")
        out_dir = fresh_out_dir(stem)
        res = run_one(local_video, effective_config(overrides), out_dir)
        files = result_files(stem, res, out_dir)
        if res["status"] != "ok":
            failure = work_root() / f"{stem}_FAILED.json"
            failure.write_text(json.dumps(res, indent=2))
            files.append((failure, f"{stem}/FAILED.json"))
        status(stage="uploading")
        if stream is not None:
            stream.publish(files, Path(path).name,
                           {"status": res["status"], "error": res.get("error"),
                            "files": [rel for _, rel in files]})
        else:
            api.publish(out_rid, out_branch, files, use_transaction)
        log.info("%s: %s, published %d files", stem, res["status"], len(files))
        cleanup(res)
        shutil.rmtree(out_dir, ignore_errors=True)
        return res
    finally:
        local_video.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Function mode
# --------------------------------------------------------------------------- #
@dataclass
class StartSplatInput:
    input_dataset_rid: str
    video_path: str
    output_dataset_rid: str
    branch: str
    config_overrides_json: str


@dataclass
class JobRef:
    job_id: str


@dataclass
class JobStatus:
    job_id: str
    video_path: str
    status: str      # queued | running | ok | rejected | failed
    stage: str
    last_log: str
    output_path: str
    error: str


JOBS: dict[str, dict] = {}
JOB_QUEUE: queue.Queue = queue.Queue()
JOBS_LOCK = threading.Lock()


class _LastLines(logging.Handler):
    def emit(self, record):
        with JOBS_LOCK:
            for j in JOBS.values():
                if j["status"] == "running":
                    j["last_log"] = self.format(record)[-300:]


def _worker():
    api = None
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS[job_id]
        req: StartSplatInput = job["request"]
        try:
            if api is None:
                from compute_modules.auth import RefreshingOauthToken
                tok = RefreshingOauthToken(hostname=foundry_hostname(),
                                           scope=["api:datasets-read", "api:datasets-write"])
                api = FoundryDatasets(foundry_hostname(), tok.get_token)
            with JOBS_LOCK:
                job["status"] = "running"

            def set_stage(stage):
                with JOBS_LOCK:
                    job["stage"] = stage

            overrides = json.loads(req.config_overrides_json) if req.config_overrides_json else None
            res = process_remote_video(api, req.input_dataset_rid, req.branch or "master",
                                       req.video_path, req.output_dataset_rid,
                                       req.branch or "master", overrides, use_transaction=True,
                                       status_cb=lambda stage: set_stage(stage))
            stem = Path(req.video_path).stem
            with JOBS_LOCK:
                job["status"] = res["status"]
                job["stage"] = "done"
                job["error"] = res.get("error") or ""
                job["output_path"] = f"{stem}/{stem}.ply" if res["status"] == "ok" else ""
        except Exception as e:
            log.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())
            with JOBS_LOCK:
                job["status"], job["error"] = "failed", str(e)[:2000]


def _status(job_id: str) -> JobStatus:
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return JobStatus(job_id, "", "unknown", "", "", "", "no such job (module restarted?)")
        return JobStatus(job_id, j["request"].video_path, j["status"], j["stage"],
                         j.get("last_log", ""), j.get("output_path", ""), j.get("error", ""))


def function_mode():
    from compute_modules import start_compute_module
    from compute_modules.annotations import function

    @function
    def start_splat(context, event: StartSplatInput) -> JobStatus:
        """Queue a video from a dataset for splatting. Returns immediately with a job id."""
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"request": event, "status": "queued", "stage": "queued",
                            "last_log": "", "output_path": "", "error": ""}
        JOB_QUEUE.put(job_id)
        return _status(job_id)

    @function
    def splat_status(context, event: JobRef) -> JobStatus:
        """Current status of a splat job."""
        return _status(event.job_id)

    @function
    def list_splat_jobs(context, event: JobRef) -> str:
        """All jobs known to this replica as JSON (job_id is ignored)."""
        with JOBS_LOCK:
            ids = list(JOBS)
        return json.dumps([_status(i).__dict__ for i in ids])

    setup_pipeline_logging()
    h = _LastLines()
    h.setFormatter(logging.Formatter("%(message)s"))
    sp.log.addHandler(h)
    threading.Thread(target=gpu_self_check, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()
    start_compute_module()


# --------------------------------------------------------------------------- #
# Local mode
# --------------------------------------------------------------------------- #
def local_mode():
    src = Path(os.environ.get("SPLAT_LOCAL_INPUT") or ENV.get("input", "/input"))
    dst = Path(os.environ.get("SPLAT_LOCAL_OUTPUT") or ENV.get("output", "/output"))
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        raise RuntimeError(f"Input folder {src} not found: put your 360 videos there")
    gpu_self_check()
    videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    log.info("Local mode: %d videos in %s -> %s", len(videos), src, dst)
    cfg_path = effective_config()
    for v in videos:
        if (dst / v.stem / f"{v.stem}.ply").exists() and os.environ.get("SPLAT_FORCE") != "1":
            log.info("%s: already done (%s), skipping. SPLAT_FORCE=1 redoes it",
                     v.name, dst / v.stem)
            continue
        out_dir = fresh_out_dir(v.stem)
        res = run_one(v, cfg_path, out_dir)
        for local, rel in result_files(v.stem, res, out_dir):
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, target)
        log.info("%s: %s -> %s", v.name, res["status"], dst / v.stem)
        cleanup(res)


ENV: dict = {}


def detect_environment(env=os.environ, root: Path = Path("/")) -> dict:
    """Work out where we're running. Uses only signals Foundry documents for each mode."""
    forced = (env.get("SPLAT_MODE") or "auto").lower()
    in_container = (root / ".dockerenv").exists() or (root / "input").is_dir()
    if env.get("BUILD2_TOKEN") and env.get("RESOURCE_ALIAS_MAP"):
        where, mode, why = "foundry", "pipeline", "BUILD2_TOKEN and RESOURCE_ALIAS_MAP are set"
    elif env.get("GET_JOB_URI") or env.get("MODULE_AUTH_TOKEN"):
        where, mode, why = "foundry", "function", "compute module job endpoints are set"
    elif in_container:
        where, mode, why = "container", "local", "running in a container outside Foundry"
    else:
        where, mode, why = "pc", "local", "no container or Foundry signals"
    info = {"where": where, "mode": mode, "reason": why}
    if forced != "auto":
        if forced not in ("pipeline", "function", "local"):
            raise SystemExit(f"Unknown SPLAT_MODE '{forced}' (use auto, pipeline, function, local)")
        info.update(mode=forced, reason=f"SPLAT_MODE={forced} (detected {where}/{mode})")
    if where == "pc":
        info.update(input=str(APP_DIR / "raw"), output=str(APP_DIR / "out"),
                    work=str(APP_DIR / "work"))
    else:
        info.update(input="/input", output="/output", work="/work")
    return info


def main():
    ENV.update(detect_environment())
    log.info("splat pipeline starting: %s mode on %s (%s)", ENV["mode"], ENV["where"], ENV["reason"])
    if "--detect" in sys.argv:
        print(json.dumps(ENV, indent=2))
        return
    {"pipeline": pipeline_mode, "function": function_mode, "local": local_mode}[ENV["mode"]]()


if __name__ == "__main__":
    main()
