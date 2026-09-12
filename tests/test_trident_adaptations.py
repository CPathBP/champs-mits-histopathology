"""The runtime adaptations register the in-domain encoder and satisfy the timm check."""

from __future__ import annotations

import pytest

trident_load = pytest.importorskip("trident.patch_encoder_models.load")

from champs_pipeline.encoders import trident_adaptations  # noqa: E402


def test_registry_gains_champs_mits():
    registry = trident_adaptations.apply()
    assert registry["champs_mits"] is trident_adaptations.ChampsMitsInferenceEncoder
    assert registry is trident_load.encoder_registry


def test_apply_is_idempotent():
    trident_adaptations.apply()
    first = trident_load.HOptimus0InferenceEncoder._build
    trident_adaptations.apply()
    assert trident_load.HOptimus0InferenceEncoder._build is first


def test_timm_version_is_asserted_only_during_build():
    import timm

    trident_adaptations.apply()
    installed = timm.__version__
    seen = {}

    def fake_build(self):
        seen["version"] = timm.__version__
        return None, None, None

    wrapped = trident_adaptations._assert_timm_version_during(fake_build)
    wrapped(object())
    assert seen["version"] == trident_adaptations.TIMM_VERSION_ASSERTED
    assert timm.__version__ == installed


def test_champs_mits_requires_weights():
    trident_adaptations.apply()
    with pytest.raises(ValueError, match="weights_path"):
        trident_adaptations.ChampsMitsInferenceEncoder()
