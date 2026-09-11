"""Verify the analysis environment.

Checks the interpreter and core package versions against the pinned values,
that Trident imports at the pinned version, that ``pip check`` reports only
the declared inconsistencies listed in EXTERNALS.md, and, when ``DATA_ROOT``
is set, that the primary data files match ``checksums.sha256``.

Exit status is non-zero on any failure. Run through ``make env-check``.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent

PINNED = {
    "torch": "2.10.0+cu128",
    "pytorch-lightning": "2.5.6",
    "timm": "1.0.27",
    "pylance": "4.0.1",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "statsmodels": "0.15.0",
    "trident": "0.2.3",
}
PYTHON = "3.10.19"

#: ``pip check`` lines that are expected and documented in EXTERNALS.md.
EXPECTED_PIP_CHECK = (
    "opencv-python-headless 4.13.0.92 has requirement numpy>=2",
    "trident 0.2.3 requires ipywidgets",
    "trident 0.2.3 requires opencv-python",
    "trident 0.2.3 has requirement timm==0.9.16",
    "trident 0.2.3 requires einops-exts",
)


def fail(msg: str, failures: list[str]) -> None:
    failures.append(msg)
    print(f"FAIL  {msg}")


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def main() -> int:
    failures: list[str] = []

    v = ".".join(map(str, sys.version_info[:3]))
    (ok if v == PYTHON else lambda m: fail(m, failures))(f"python {v} (pinned {PYTHON})")

    for dist, want in PINNED.items():
        try:
            have = metadata.version(dist)
        except metadata.PackageNotFoundError:
            fail(f"{dist} not installed (pinned {want})", failures)
            continue
        if have == want:
            ok(f"{dist} {have}")
        else:
            fail(f"{dist} {have} (pinned {want})", failures)

    try:
        import trident  # noqa: F401
        ok("trident imports")
    except Exception as e:  # pragma: no cover
        fail(f"trident import: {e}", failures)

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    unexpected = [
        line for line in proc.stdout.splitlines()
        if line.strip() and not any(line.startswith(p) for p in EXPECTED_PIP_CHECK)
        and not line.startswith("champs-pipeline")
    ]
    if unexpected:
        for line in unexpected:
            fail(f"pip check: {line}", failures)
    else:
        ok("pip check: only the documented inconsistencies")

    data_root = os.environ.get("DATA_ROOT")
    if data_root:
        root = Path(data_root)
        n = 0
        for line in (HERE / "checksums.sha256").read_text().splitlines():
            digest, _, rel = line.partition("  ")
            p = root / rel
            if not p.exists():
                fail(f"missing {rel}", failures)
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h == digest:
                n += 1
            else:
                fail(f"checksum mismatch {rel}", failures)
        ok(f"{n} primary files match checksums.sha256")
    else:
        print("skip  DATA_ROOT not set; primary-data checksums not verified")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
