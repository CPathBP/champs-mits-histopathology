"""The run record of an extraction: which schema, prompt and settings produced a directory.

The extractor writes ``run_meta.json`` into its output directory when it
starts. The record fingerprints the schema and the prompt and stores the
inference settings, so that a later invocation into the same directory
with another schema or prompt is refused, and so that the finaliser can
verify it is given the schema and prompt the records were extracted with.
"""

import hashlib
import json
import time
from pathlib import Path

RUN_META_FILENAME = "run_meta.json"


def file_sha(path):
    """The first 16 hex digits of the SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def text_sha(text):
    """The first 16 hex digits of the SHA-256 of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_run_meta(out_dir):
    """The run record of a directory, or ``None`` when there is none."""
    path = Path(out_dir) / RUN_META_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_or_check_run_meta(out_dir, *, model, schema, schema_path, prompt_path, settings=None):
    """Write the run record on the first run; on later runs verify it matches.

    ``settings`` is the dict of inference settings the extractor uses
    (engine build, context length, output cap, decoding constraint, the
    fingerprint of the rendered prompt); it is stored with the first
    record. Raises ``SystemExit`` when the directory records another schema
    or prompt, because the directory would then mix two extraction setups.
    """
    out_dir = Path(out_dir)
    record = {
        "model": model,
        "schema_path": str(schema_path),
        "schema_version": schema.schema_version,
        "schema_sha": schema.schema_sha,
        "prompt_template": str(prompt_path),
        "prompt_sha": file_sha(prompt_path),
        "settings": settings or {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    existing = load_run_meta(out_dir)
    if existing is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / RUN_META_FILENAME).write_text(json.dumps(record, indent=2))
        return record
    mismatches = [
        f"  {key}: recorded {existing.get(key)!r}, current {record[key]!r}"
        for key in ("schema_sha", "prompt_sha")
        if existing.get(key) != record[key]
    ]
    if mismatches:
        raise SystemExit(
            f"{out_dir / RUN_META_FILENAME} records a different extraction setup:\n"
            + "\n".join(mismatches)
            + "\nUse a new output directory for a new schema or prompt version."
        )
    return existing
