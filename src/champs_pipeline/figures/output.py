"""Output of figures and tables together with their source data and provenance.

Every display is saved to its own directory, ``<out_dir>/<name>/``. The name
states whether the display belongs to the main text or to the supplementary
information, for example ``main_table_cohort``.

A figure is saved as ``<name>.pdf`` (vector) and ``<name>.png`` (300 dpi).
Its source data are saved as ``<name>_source_data.csv`` or, for a figure
with several parts, as ``<name>_source_data_<part>.csv``. A table is saved
as ``<name>.tex`` and ``<name>.csv``. Each display is accompanied by
``<name>.provenance.json``, which records the generating command, the
repository commit, whether the working tree differed from that commit,
whether the display is a draft from unchecked predictions, and the SHA-256
digest of every input file.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from champs_pipeline.eval.run_record import git_state, now
from champs_pipeline.figures import latex, style
from champs_pipeline.figures.check import layout_problems, problems

REPOSITORY = Path(__file__).resolve().parents[3]
CHUNK_BYTES = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(value):
    """Write a command-line value without the directory layout of the local system.

    A path inside the repository is written relative to the repository, any other
    existing path as its last two components (for example ``cohort/funnel.json``),
    and every other value unchanged.
    """
    path = Path(value)
    if not path.exists():
        return str(value)
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY):
        return str(resolved.relative_to(REPOSITORY))
    return str(Path(*resolved.parts[-2:]))


def write_provenance(out_dir, name, inputs, draft=False):
    """Write ``<name>.provenance.json`` to ``out_dir`` and return the record."""
    commit, dirty = git_state(REPOSITORY)
    record = {
        "display": name,
        "command": [portable(value) for value in sys.argv],
        "created": now(),
        "git_commit": commit,
        "git_dirty": dirty,
        "draft": draft,
        "matplotlib": matplotlib.__version__,
        "inputs": [{"path": portable(path), "sha256": sha256(path)} for path in inputs],
    }
    path = Path(out_dir) / f"{name}.provenance.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def source_data_path(out_dir, name, part):
    if part:
        return Path(out_dir) / f"{name}_source_data_{part}.csv"
    return Path(out_dir) / f"{name}_source_data.csv"


def save_figure(fig, out_dir, name, source_data, inputs, draft=False, data_axes=(),
                clear_axes=False):
    """Verify the figure against the journal requirements and save it with its source data.

    ``source_data`` maps each part letter to the data frame shown in that part;
    a figure with a single part uses the key ``""``. ``inputs`` lists the files
    from which the data were read. ``draft`` marks a figure made from unchecked
    predictions in its provenance. ``data_axes`` and ``clear_axes`` set the
    text-layout check (``check.layout_problems``). If the figure violates a
    requirement, ``ValueError`` is raised and no file is written.
    """
    found = problems(fig) + layout_problems(fig, data_axes, clear_axes)
    if found:
        raise ValueError(f"{name} violates the figure requirements:\n  " + "\n  ".join(found))
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    # The creation date is omitted so that identical inputs yield an identical PDF.
    fig.savefig(out_dir / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out_dir / f"{name}.png", dpi=style.PNG_DPI)
    for part, frame in source_data.items():
        frame.to_csv(source_data_path(out_dir, name, part), index=False)
    write_provenance(out_dir, name, inputs, draft)
    plt.close(fig)


def write_table(table, out_dir, name, inputs, align=None, group_column=None, header_width=None,
                long=False, draft=False):
    """Save the table as LaTeX and as CSV, together with its provenance.

    ``align``, ``group_column``, ``header_width`` and ``long`` are passed to
    ``latex.tabular``. ``draft`` marks a table made from unchecked predictions
    in its provenance.
    """
    tex = latex.tabular(table, align=align, group_column=group_column,
                        header_width=header_width, long=long)
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.tex").write_text(tex)
    table.to_csv(out_dir / f"{name}.csv", index=False)
    write_provenance(out_dir, name, inputs, draft)
