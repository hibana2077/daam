"""Self-check: python test_daam.py  (random weights, CPU, no downloads)."""

import timm
import torch
from PIL import Image

from daam import DAAM, heatmap, overlay

torch.manual_seed(0)
x = torch.randn(2, 3, 224, 224)

# Supervised: [CLS] decomposition, Eq. 7-12.
model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
fused = model.blocks[0].attn.fused_attn
with DAAM(model) as daam:
    r = daam(x, target=[3, 7])
    assert r.block_maps.shape == (2, 12, 14, 14) and r.maps.shape == (2, 12, 14, 14)
    assert r.target.tolist() == [3, 7] and r.output.shape == (2, 1000)
    assert torch.isfinite(r.block_maps).all() and (r.block_maps >= 0).all()
    assert torch.allclose(r.maps[:, -1], r.block_maps.sum(1))
    # Eq. 7: summing S_b = a_b ⊗ V_b over tokens must give T_b (the input of attn.proj).
    T = {}
    h = model.blocks[0].attn.proj.register_forward_hook(lambda m, i, o: T.update(t=i[0].detach()))
    daam(x[:1])
    h.remove()
    a, v = daam._a[0], daam._v[0]  # (1, H, N, N), (1, N, C)
    s = torch.einsum("bhn,bnhd->bnhd", a[:, :, 0], v.view(1, v.shape[1], a.shape[1], -1)).sum(1).reshape(1, -1)
    assert torch.allclose(s, T["t"][:, 0], atol=1e-5), "S_b does not sum to T_b"
    assert not model.blocks[0].attn.fused_attn
assert model.blocks[0].attn.fused_attn == fused, "fused_attn not restored"

# Explicit mode: feature decomposition on a model that still has a (possibly unused) head.
with DAAM(model) as daam:
    r = daam(x[:1], mode="feature")
    assert r.target is None and r.output.shape == (1, 192)
    assert torch.allclose(r.output, model.forward_head(model.forward_features(x[:1]), pre_logits=True))

# Self-supervised: no head, dimension-wise weight (Eq. 13), self and reference targets.
model.reset_classifier(0)
with DAAM(model) as daam:
    r = daam(x[:1])
    assert r.target is None and r.output.shape == (1, 192) and r.block_maps.shape == (1, 12, 14, 14)
    r2 = daam(x[:1], target=torch.randn(5, 192))
    assert r2.block_maps.shape == (1, 12, 14, 14) and torch.isfinite(r2.block_maps).all()
    try:
        daam(x[:1], mode="class")
        raise AssertionError("mode='class' without a head should fail")
    except ValueError:
        pass

# Average pooling + register tokens (EVA / DINOv3-style models).
model = timm.create_model("eva02_tiny_patch14_224", pretrained=False)
with DAAM(model) as daam:
    r = daam(x[:1, :, :224, :224])
    assert r.block_maps.shape == (1, 12, 16, 16)

assert heatmap(r.maps[0]).max() == 255 and heatmap(r.maps[0, -1]).shape == (1, 16, 16)
hm = heatmap(r.maps[0], size=(224, 224))
assert hm.shape == (12, 224, 224) and hm.dtype.name == "uint8" and hm[-1].max() >= 250
assert overlay(Image.new("RGB", (64, 64)), hm[-1]).size == (64, 64)
print("ok")
