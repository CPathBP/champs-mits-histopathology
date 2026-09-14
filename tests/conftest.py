import sys
from pathlib import Path

# The scripts are plain files; make them importable by module name.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for group in ("slides", "reports", "cohort"):
    sys.path.insert(0, str(SCRIPTS / group))

import pytest


@pytest.fixture(scope="session")
def schema():
    from champs_pipeline.data_prep.morphology_labels import load_schema

    return load_schema(SCRIPTS.parent / "configs" / "extraction" / "morphology_schema_v4_4.yaml")


@pytest.fixture(scope="session")
def schema_v45():
    from champs_pipeline.data_prep.morphology_labels import load_schema

    return load_schema(SCRIPTS.parent / "configs" / "extraction" / "morphology_schema_v4_5.yaml")
