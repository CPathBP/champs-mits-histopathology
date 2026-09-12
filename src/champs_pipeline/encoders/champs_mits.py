"""The in-domain encoder.

A DINOv2 ViT-g/14 backbone (four register tokens, SwiGLU feed-forward,
LayerScale) warm-started from H-optimus-0 and extended by identity-initialised
transformer blocks during self-supervised training on CHAMPS MITS tiles
(``scripts/slides/ssl``). Inference uses the teacher backbone, the
H-optimus-0 preprocessing, and the class token, so the features are directly
comparable with the H-optimus-0 features extracted by Trident.

The checkpoint may be a raw backbone state dict, ``{"model": state_dict}``
(the warm-start file), or a DINOv2 training checkpoint with the teacher under
``teacher``.
"""

from __future__ import annotations

import copy
import re

import torch
import torch.nn as nn

EMBED_DIM = 1536
INIT_VALUES = 1e-5

# H-optimus-0 normalisation, as applied by Trident for that encoder.
HOPT_MEAN = (0.707223, 0.578729, 0.703617)
HOPT_STD = (0.211883, 0.230117, 0.177517)


def build_backbone():
    """The DINOv2 ViT-g/14 in the layout of the warm-start checkpoint."""
    from dinov2.models.vision_transformer import vit_giant2

    return vit_giant2(
        patch_size=14,
        img_size=224,
        ffn_layer="swiglufused",
        block_chunks=0,
        num_register_tokens=4,
        init_values=INIT_VALUES,
    )


def backbone_state_dict(ckpt: dict) -> dict:
    """The backbone tensors from any of the supported checkpoint layouts."""
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    for key in ("model", "teacher", "student"):
        if key in ckpt:
            sub = ckpt[key]
            if any(k.startswith("backbone.") for k in sub):
                prefix = "backbone."
                return {k[len(prefix):]: v for k, v in sub.items() if k.startswith(prefix)}
            return sub
    raise KeyError(f"no backbone in checkpoint keys: {list(ckpt)[:6]}")


def _n_blocks(sd: dict) -> int:
    idxs = [int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))]
    return max(idxs) + 1 if idxs else 0


class ChampsMitsEncoder(nn.Module):
    """Backbone of the in-domain encoder; ``forward`` returns the class token."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.backbone = build_backbone()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = backbone_state_dict(ckpt)
        # A block-expanded checkpoint carries more blocks than the base
        # network. Append copies so the trained blocks load instead of being
        # dropped silently by a non-strict load.
        n_ckpt, n_have = _n_blocks(sd), len(self.backbone.blocks)
        for _ in range(n_ckpt - n_have):
            self.backbone.blocks.append(copy.deepcopy(self.backbone.blocks[-1]))
        self.backbone.n_blocks = len(self.backbone.blocks)
        missing, _unexpected = self.backbone.load_state_dict(sd, strict=False)
        real_missing = [m for m in missing if m != "mask_token"]
        if real_missing:
            raise RuntimeError(f"checkpoint lacks backbone parameters: {real_missing[:6]}")
        self.backbone.eval()
        self.enc_name = "champs_mits"
        self.precision = torch.float16

    @torch.inference_mode()
    def forward(self, x):
        return self.backbone.forward_features(x)["x_norm_clstoken"]


def eval_transform():
    """The H-optimus-0 preprocessing."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=HOPT_MEAN, std=HOPT_STD),
    ])
