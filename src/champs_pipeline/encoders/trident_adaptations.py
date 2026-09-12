"""Runtime adaptations of Trident.

Trident v0.2.3 is installed unmodified; its licence does not permit
redistributing a modified copy. Two things the study needs are applied to
the imported package by :func:`apply`:

1. The in-domain encoder is added to the patch-encoder registry under the
   name ``champs_mits``, so it runs through the same extraction path as the
   public encoders.
2. The H-optimus-0 and H-optimus-1 loaders assert ``timm == 0.9.16``. The
   assertion is satisfied while those loaders build their model. The
   architecture they build, ``vit_giant_patch14_reg4_dinov2``, produces the
   same features under timm 1.0.27, which was verified against the stored
   features of the study.

Call :func:`apply` before :func:`trident.patch_encoder_models.encoder_factory`.
"""

from __future__ import annotations

import functools

import torch
from trident.patch_encoder_models import load as _trident_load

TIMM_VERSION_ASSERTED = "0.9.16"


class ChampsMitsInferenceEncoder(_trident_load.BasePatchEncoder):
    """Trident wrapper of the in-domain encoder. Requires ``weights_path``."""

    def _build(self):
        from champs_pipeline.encoders.champs_mits import ChampsMitsEncoder, eval_transform

        self.enc_name = "champs_mits"
        if not self.weights_path:
            raise ValueError("champs_mits needs weights_path=<trained checkpoint>")
        self.ensure_valid_weights_path(self.weights_path)
        return ChampsMitsEncoder(self.weights_path), eval_transform(), torch.float16


def _assert_timm_version_during(build):
    @functools.wraps(build)
    def wrapper(self, *args, **kwargs):
        import timm

        installed = timm.__version__
        timm.__version__ = TIMM_VERSION_ASSERTED
        try:
            return build(self, *args, **kwargs)
        finally:
            timm.__version__ = installed

    wrapper.adapted = True
    return wrapper


def apply() -> dict:
    """Apply both adaptations once; returns Trident's encoder registry."""
    registry = _trident_load.encoder_registry
    registry.setdefault("champs_mits", ChampsMitsInferenceEncoder)
    for cls in (_trident_load.HOptimus0InferenceEncoder, _trident_load.HOptimus1InferenceEncoder):
        if not getattr(cls._build, "adapted", False):
            cls._build = _assert_timm_version_during(cls._build)
    return registry
