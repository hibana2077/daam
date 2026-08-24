"""Dynamic Accumulated Attention Map (DAAM) for timm Vision Transformers.

Liao, Gao, Zhang. "Dynamic Accumulated Attention Map for Interpreting Evolution
of Decision-making in Vision Transformer", Pattern Recognition 2025.

Per block b (Eq. 7-9):  S_b = a_b ⊗ V_b,  C_b = ∂Y/∂T_b,  L_b = Σ_d ReLU(C_b)_d · S_b[:, d]
Accumulated map (Eq. 11-12):  L_{b,DAAM} = Σ_{i<=b} L_i

Y is either a class score (supervised, Eq. 8) or the inner product between the
pooled feature and a dimension-wise importance weight (self-supervised, Eq. 13).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = ["DAAM", "DAAMResult", "heatmap", "overlay"]


@dataclass
class DAAMResult:
    output: torch.Tensor  # (B, C) logits, or (B, D) pooled features when the model has no head
    target: torch.Tensor | None  # (B,) explained class index; None in feature mode
    block_maps: torch.Tensor  # (B, L, h, w) per-block maps L_b on the patch grid

    @property
    def maps(self):
        """Accumulated maps L_{b,DAAM} for every block, (B, L, h, w)."""
        return self.block_maps.cumsum(1)


class DAAM:
    """Hooks a timm ViT (``model.blocks[*].attn``) and computes DAAM maps.

    ``mode='class'`` (default when the model has a head): ``target`` is a class index (default: argmax).
    ``mode='feature'`` (default when ``num_classes == 0``; pass explicitly to explain a backbone's
    pre-logits feature and ignore its head): ``target`` is an optional ``(K, D)`` tensor of
    reference features (memory-bank neighbours, prototypes, ...). Default: the image's own
    feature, i.e. w_d = f̂_d².
    Pooling: ``global_pool='token'`` decomposes the [CLS] token (paper); ``'avg'``
    decomposes the mean patch token the same way.
    """

    def __init__(self, model):
        self.model = model.eval()
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))
        self.pool = getattr(model, "global_pool", "token") or "token"
        if self.pool not in ("token", "avg"):
            raise ValueError(f"global_pool={self.pool!r} is not supported (need 'token' or 'avg').")
        self._a, self._v, self._t, self._handles, self._fused = [], [], [], [], []
        for i, block in enumerate(model.blocks):
            attn = getattr(block, "attn", None)
            if not all(hasattr(attn, n) for n in ("attn_drop", "proj", "num_heads")):
                raise ValueError(f"blocks[{i}].attn is not a timm ViT attention module.")
            # timm only materialises the attention matrix on the non-fused path.
            self._fused.append((attn, getattr(attn, "fused_attn", None)))
            attn.fused_attn = False
            norm = getattr(attn, "norm", None)
            t_layer = norm if isinstance(norm, torch.nn.Module) else attn.proj  # its input is T_b = A_b V_b
            self._handles += [
                attn.register_forward_pre_hook(lambda m, args: self._v.append(_value(m, args[0]))),
                attn.attn_drop.register_forward_hook(lambda m, i, o: self._a.append(o.detach())),
                t_layer.register_forward_hook(lambda m, i, o: self._t.append(i[0])),
            ]

    def __call__(self, x, target=None, mode=None):
        """mode: 'class' (Eq. 8), 'feature' (Eq. 13, pre-logits feature regardless of head),
        or None = 'feature' iff ``model.num_classes == 0``."""
        mode = mode or ("feature" if getattr(self.model, "num_classes", 1) == 0 else "class")
        if mode == "class" and getattr(self.model, "num_classes", 1) == 0:
            raise ValueError("mode='class' needs a classifier head (model.num_classes > 0).")
        self._a, self._v, self._t = [], [], []
        x = x.detach().to(next(self.model.parameters()).device).requires_grad_(True)
        with torch.enable_grad():
            if mode == "class":
                out = self.model(x)
            else:
                out = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
            w, target = self._weight(out, target, mode)
            grads = torch.autograd.grad(out, self._t, grad_outputs=w)
        self._t = []  # release the graph
        rows = slice(0, 1) if self.pool == "token" else slice(self.prefix, None)
        maps = []
        for a, v, g in zip(self._a, self._v, grads):
            B, N, C = v.shape
            H = a.shape[1]
            g = g[:, rows].relu().reshape(B, -1, H, C // H)  # ReLU(C_b), (B, R, H, d)
            coef = torch.einsum("bhrn,brhd->bnhd", a[:, :, rows], g)  # a_b ⊗ ReLU(C_b) routed to each token
            maps.append((coef * v.view(B, N, H, -1)).sum((2, 3))[:, self.prefix :].relu())  # Eq. 9
        h, w_ = self._grid(x.shape[-2:], maps[0].shape[1])
        return DAAMResult(out.detach(), target, torch.stack(maps, 1).view(B, -1, h, w_))

    def _weight(self, out, target, mode):
        """grad_outputs for ∂Y/∂T: one-hot class (Eq. 8) or dimension-wise weight (Eq. 13)."""
        if mode == "feature":
            f = F.normalize(out, dim=-1)
            if target is None:
                return f * f, None
            z = F.normalize(torch.as_tensor(target, device=out.device, dtype=out.dtype).view(-1, out.shape[-1]), dim=-1)
            prod = f[:, None] * z[None]  # (B, K, D); each row sums to cos(f, z_k)
            return (prod / prod.sum(-1, keepdim=True).abs().clamp(min=1e-10)).mean(1), None
        if target is None:
            target = out.argmax(-1)
        target = torch.as_tensor(target, device=out.device).view(-1).expand(len(out))
        return F.one_hot(target, out.shape[-1]).to(out.dtype), target

    def _grid(self, size, n):
        pe = getattr(self.model, "patch_embed", None)
        h, w = pe.dynamic_feat_size(tuple(size)) if hasattr(pe, "dynamic_feat_size") else (int(n**0.5),) * 2
        if h * w != n:
            raise ValueError(f"Cannot reshape {n} patch tokens into a {h}x{w} grid.")
        return h, w

    def close(self):
        for handle in self._handles:
            handle.remove()
        for attn, fused in self._fused:
            attn.fused_attn = fused
        self._handles, self._fused = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _value(attn, x):
    """V_b from the attention input, covering fused qkv (+ separate biases) and split projections."""
    qkv = getattr(attn, "qkv", None)
    if qkv is None:
        return attn.v_proj(x).detach()
    d = qkv.weight.shape[0] // 3
    bias = qkv.bias[-d:] if qkv.bias is not None else getattr(attn, "v_bias", None)
    return F.linear(x, qkv.weight[-d:], bias).detach()


def heatmap(maps, size=None):
    """Render accumulated maps as uint8 images, as in the official code.

    ``maps``: (L, h, w) for one image (``result.maps[i]``) or a single (h, w) map.
    Every map is scaled by the final block's maximum (so earlier blocks look dimmer),
    passed through sigmoid(5x) - 0.5, min-max normalised, then bilinearly resized.
    Returns (L, H, W) uint8.
    """
    m = torch.as_tensor(maps).detach().float().cpu().clamp(min=0)
    m = m[None] if m.ndim == 2 else m
    m = torch.sigmoid(5 * m / m[-1].max().clamp(min=1e-10)) - 0.5
    lo, hi = m.flatten(1).min(1).values[:, None, None], m.flatten(1).max(1).values[:, None, None]
    m = (m - lo) / (hi - lo).clamp(min=1e-10)
    if size is not None:
        m = F.interpolate(m[None], size=tuple(size), mode="bilinear", align_corners=False)[0]
    return (m * 255).round().to(torch.uint8).numpy()


def overlay(image, hm, alpha=0.4):
    """Blend a uint8 (H, W) heatmap (jet-like colormap) onto a PIL image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    h = np.asarray(Image.fromarray(np.asarray(hm, dtype=np.uint8)).resize(image.size), dtype=np.float32) / 255
    color = np.stack([1.5 * h, 1.5 * (1 - np.abs(h - 0.5) * 2), 1.5 * (1 - h)], -1).clip(0, 1) * 255
    out = (1 - alpha) * np.asarray(image, dtype=np.float32) + alpha * color
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
