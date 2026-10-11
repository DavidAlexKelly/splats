"""
Foundry Python transform: rebuild the splat result files from the stream of chunk records
written by the compute module (see splat_stream.py for the record format).

Put this file in a Python transforms Code Repository (e.g. src/myproject/datasets/),
set the two paths below, commit, and build the output dataset. Schedule it to rebuild
when the stream updates (or every hour) to pick up new splats automatically.

Rules (identical to splat_stream.rebuild):
  * chunks may arrive in any order and may be duplicated -> sorted + de-duplicated
  * a file is only rebuilt when every chunk of a run is present
  * if a video was processed more than once, the newest complete run wins
  * every file is checked against its SHA-256 before it is written to the output
"""
import base64
import hashlib
import json
import shutil
import tempfile

from pyspark.sql import Window
from pyspark.sql import functions as F
from transforms.api import Input, Output, transform

STREAM_PATH = "/YOUR/PROJECT/PATH/splat_stream"       # the compute module's output stream
OUTPUT_PATH = "/YOUR/PROJECT/PATH/splat_files"        # dataset that will hold the real files


@transform(
    out=Output(OUTPUT_PATH),
    records=Input(STREAM_PATH),
)
def compute(out, records):
    chunks = records.dataframe().filter(F.col("record_type") == "chunk")

    runs = (chunks
            .groupBy("file_path", "run_id", "video", "chunk_count", "total_bytes", "sha256")
            .agg(F.countDistinct("chunk_index").alias("have"),
                 F.max("created_at").alias("latest"))
            .filter(F.col("have") == F.col("chunk_count")))
    newest = Window.partitionBy("file_path").orderBy(F.col("latest").desc())
    chosen = (runs.withColumn("rn", F.row_number().over(newest))
              .filter(F.col("rn") == 1).collect())

    manifest = []
    fs = out.filesystem()
    for g in chosen:
        rows = (chunks
                .filter((F.col("file_path") == g.file_path) & (F.col("run_id") == g.run_id))
                .dropDuplicates(["chunk_index"])
                .orderBy("chunk_index")
                .select("data_b64")
                .toLocalIterator())
        h, size = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile() as tmp:
            for r in rows:  # streamed: a large .ply never has to fit in memory at once
                data = base64.b64decode(r.data_b64)
                h.update(data)
                size += len(data)
                tmp.write(data)
            ok = h.hexdigest() == g.sha256 and size == int(g.total_bytes)
            if ok:
                tmp.flush()
                tmp.seek(0)
                with fs.open(g.file_path, "wb") as dst:
                    shutil.copyfileobj(tmp, dst, length=8 << 20)
        manifest.append({"file_path": g.file_path, "video": g.video, "run_id": g.run_id,
                         "bytes": size, "verified": ok})

    with fs.open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
