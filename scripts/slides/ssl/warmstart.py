"""Convert H-optimus-0 into a DINOv2 warm-start checkpoint.

H-optimus-0 is the timm model ``vit_giant_patch14_reg4_dinov2``, which is
DINOv2's ViT-g/14 with SwiGLU, four register tokens, and LayerScale, so the
conversion renames keys: ``reg_token`` to ``register_tokens``, ``pos_embed``
gains a zero class-token position at index 0, and ``mlp.fc1``/``fc2`` become
``mlp.w12``/``w3``. ``--validate`` checks the mapping by forward equivalence
with random weights and reports whether the fused SwiGLU halves need swapping.
Output: ``{"model": state_dict}`` for ``student.pretrained_weights``.
"""

import argparse
import sys
from pathlib import Path

import torch

from champs_pipeline.encoders.champs_mits import INIT_VALUES, build_backbone

TIMM_KWARGS = {
    "num_classes": 0,
    "img_size": 224,
    "global_pool": "token",
    "init_values": INIT_VALUES,
    "dynamic_img_size": False,
}


def timm_to_dinov2(sd, swiglu_swap=False):
    out = {}
    for k, v in sd.items():
        if k == "reg_token":
            out["register_tokens"] = v
        elif k == "pos_embed":
            out["pos_embed"] = torch.cat([torch.zeros(1, 1, v.shape[-1], dtype=v.dtype), v], dim=1)
        elif ".mlp.fc1." in k:
            if swiglu_swap:
                a, b = v.chunk(2, dim=0)
                v = torch.cat([b, a], dim=0)
            out[k.replace(".mlp.fc1.", ".mlp.w12.")] = v
        elif ".mlp.fc2." in k:
            out[k.replace(".mlp.fc2.", ".mlp.w3.")] = v
        else:
            out[k] = v
    return out


def load(backbone, sd):
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "mask_token"]
    if missing or unexpected:
        raise RuntimeError(f"mapping incomplete: missing {missing[:4]} unexpected {unexpected[:4]}")


@torch.inference_mode()
def validate():
    import timm

    torch.manual_seed(0)
    ref_model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, **TIMM_KWARGS)
    ref_model = ref_model.eval()
    x = torch.randn(1, 3, 224, 224)
    ref = ref_model(x)
    for swap in (False, True):
        backbone = build_backbone().eval()
        load(backbone, timm_to_dinov2(ref_model.state_dict(), swiglu_swap=swap))
        diff = (backbone.forward_features(x)["x_norm_clstoken"] - ref).abs().max().item()
        print(f"swiglu_swap={swap}: max abs difference of the class token {diff:.3e}")
        if diff < 1e-3:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="Local H-optimus-0 pytorch_model.bin (default: the Hugging Face cache).")
    ap.add_argument("--out", default="hoptimus0_dinov2_warmstart.pth")
    ap.add_argument("--swiglu-swap", action="store_true")
    args = ap.parse_args()
    if args.validate:
        sys.exit(0 if validate() else 1)
    import timm

    if args.weights:
        model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, **TIMM_KWARGS)
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=True)
    else:
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **TIMM_KWARGS)
    sd = timm_to_dinov2(model.state_dict(), swiglu_swap=args.swiglu_swap)
    load(build_backbone(), sd)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
