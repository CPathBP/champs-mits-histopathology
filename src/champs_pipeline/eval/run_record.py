"""The run record: ``run.json`` in every run directory.

Written before training starts, so an interrupted run is still
identifiable, and updated at the end with the selected checkpoint and the
status. A reader takes what a run says about itself from this file and
never from the shape of its path. Writes are atomic, so a reader never sees
half a record.
"""

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

FILENAME = "run.json"
SCHEMA_VERSION = 4
# Untracked files under these paths change what runs, so they make the tree dirty.
CODE_PATHS = ("src/", "scripts/", "configs/", "env/", "pyproject.toml")


def is_dirty(porcelain):
    """Whether ``git status --porcelain`` output shows a change to the code."""
    for line in porcelain.splitlines():
        path = line[3:]
        if not line.startswith("??") or path.startswith(CODE_PATHS):
            return True
    return False


def git_state(repo):
    """The short commit and whether the code differs from it."""
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False)
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                 "--untracked-files=all"],
                                capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None, None
    if commit.returncode != 0 or status.returncode != 0:
        return None, None
    return commit.stdout.strip() or None, is_dirty(status.stdout)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_id():
    """The Slurm job of this process as ``squeue`` names it, or ``None`` outside Slurm."""
    array_job, task = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and task:
        return f"{array_job}_{task}"
    return os.environ.get("SLURM_JOB_ID")


def _dump(run_dir, record):
    path = Path(run_dir) / FILENAME
    temporary = path.with_name(FILENAME + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def write(run_dir, repo, **fields):
    """Create the record with the code and job identity; ``status`` starts as running."""
    commit, dirty = git_state(repo)
    record = {"schema_version": SCHEMA_VERSION, "status": "running", "started": now(),
              "finished": None, "git_commit": commit, "git_dirty": dirty,
              "slurm_job_id": job_id(), "host": socket.gethostname(), **fields}
    _dump(run_dir, record)
    return record


def read(run_dir):
    """The record; ``None`` when there is none, status ``unreadable`` when it does not parse."""
    path = Path(run_dir) / FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"status": "unreadable"}


def update(run_dir, **fields):
    """Set fields on an existing record."""
    record = read(run_dir)
    if record is None or record.get("status") == "unreadable":
        raise FileNotFoundError(f"no readable {FILENAME} in {run_dir}")
    record.update(fields)
    _dump(run_dir, record)
    return record


def is_live(record):
    """Whether a running record belongs to another Slurm job that is still queued or running."""
    recorded = record.get("slurm_job_id")
    if record.get("status") != "running" or not recorded or recorded == job_id():
        return False
    try:
        queue = subprocess.run(["squeue", "-h", "-j", recorded, "-o", "%T"],
                               capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return True  # the queue cannot be asked, so the attempt is left alone
    return bool(queue.stdout.strip())
