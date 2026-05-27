import math

import torch
from torch.nn import functional as F

try:
    from timm.layers import apply_rot_embed_cat
except Exception:  # timm is imported lazily by users of create_daam.
    apply_rot_embed_cat = None


def prefix_tokens(model):
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    return 1 if hasattr(model, "cls_token") else 0


def uses_class_token(model):
    global_pool = getattr(model, "global_pool", "")
    return prefix_tokens(model) > 0 and global_pool in ("", "token", None)


def reshape_tokens(tensor, prefix=1, grid_size=None):
    tokens = tensor[:, prefix:, :] if prefix else tensor
    if grid_size is None:
        side = int(math.sqrt(tokens.shape[1]))
        grid_size = (side, side)
    if grid_size[0] * grid_size[1] != tokens.shape[1]:
        raise ValueError("Could not reshape ViT patch tokens into a spatial grid.")
    return tokens.reshape(tokens.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2)


def collect_vit_layers(model):
    if not hasattr(model, "blocks"):
        raise ValueError("Only timm ViT-style models with `model.blocks` are supported.")

    layers = []
    for index, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        proj = getattr(attn, "proj", None) if attn is not None else None
        has_qkv = has_supported_qkv(attn)
        if attn is None or proj is None or not has_qkv:
            raise ValueError(f"Block {index} is not a supported timm ViT attention block.")
        layers.append((attn, proj))
    return layers


def has_supported_qkv(attn):
    if attn is None:
        return False

    qkv = getattr(attn, "qkv", None)
    if callable(qkv):
        out_features = getattr(qkv, "out_features", None)
        num_heads = getattr(attn, "num_heads", None)
        head_dim = getattr(attn, "head_dim", None)
        if out_features and num_heads and head_dim and out_features != 3 * num_heads * head_dim:
            return False
        return True

    return all(callable(getattr(attn, name, None)) for name in ("q_proj", "k_proj", "v_proj"))


class ViTAttentionHooks:
    """Collect DAAM activations and projection-input gradients for ViT blocks."""

    def __init__(self, model):
        self.model = model
        self.prefix = prefix_tokens(model)
        self.use_class_token = uses_class_token(model)
        self.activations = []
        self.gradients = []
        self.handles = []

        for attn, proj in collect_vit_layers(model):
            self.handles.append(self._forward_hook(attn, self._save_activation))
            self.handles.append(self._forward_hook(proj, self._save_gradient(attn)))

    def clear(self):
        self.activations = []
        self.gradients = []

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x):
        self.clear()
        return self.model(x)

    def _forward_hook(self, module, fn):
        try:
            return module.register_forward_hook(fn, with_kwargs=True)
        except TypeError:
            return module.register_forward_hook(lambda m, i, o: fn(m, i, {}, o))

    def _save_activation(self, module, inputs, kwargs, output):
        x = inputs[0]
        if not torch.is_tensor(x):
            return
        activation = attention_value_product(
            module,
            x,
            prefix=self.prefix,
            use_class_token=self.use_class_token,
            rope=kwargs.get("rope"),
            attn_mask=kwargs.get("attn_mask"),
        )
        self.activations.append(activation.detach().cpu())

    def _save_gradient(self, attn):
        def hook(module, inputs, kwargs, output):
            if not inputs or not torch.is_tensor(inputs[0]) or not inputs[0].requires_grad:
                return
            inputs[0].register_hook(lambda grad: self._store_gradient(grad))

        return hook

    def _store_gradient(self, grad):
        if self.use_class_token:
            weight = grad[:, 0, :]
        elif self.prefix:
            weight = grad[:, self.prefix :, :].mean(dim=1)
        else:
            weight = grad.mean(dim=1)
        self.gradients.insert(0, weight.detach().cpu())


def attention_value_product(module, x, prefix=1, use_class_token=True, rope=None, attn_mask=None):
    q, k, v = project_qkv(module, x)
    q, k = normalize_qk(module, q, k)
    q, k = apply_rope(module, q, k, v, prefix, rope)

    attn = (q * module.scale) @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)

    drop = getattr(module, "attn_drop", None)
    if drop is not None:
        attn = drop(attn)

    focus = attn[:, :, 0] if use_class_token else attn[:, :, prefix:, :].mean(dim=2)
    values = focus.unsqueeze(-1) * v
    return values.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)


def project_qkv(module, x):
    batch, tokens, _ = x.shape
    if callable(getattr(module, "qkv", None)):
        qkv = qkv_tensor(module, x)
        head_dim = getattr(module, "head_dim", qkv.shape[-1] // (3 * module.num_heads))
        qkv = qkv.reshape(batch, tokens, 3, module.num_heads, head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    q = module.q_proj(x).reshape(batch, tokens, module.num_heads, -1).transpose(1, 2)
    k = module.k_proj(x).reshape(batch, tokens, module.num_heads, -1).transpose(1, 2)
    v = module.v_proj(x).reshape(batch, tokens, module.num_heads, -1).transpose(1, 2)
    return q, k, v


def qkv_tensor(module, x):
    q_bias = getattr(module, "q_bias", None)
    k_bias = getattr(module, "k_bias", None)
    v_bias = getattr(module, "v_bias", None)
    if q_bias is None or v_bias is None:
        return module.qkv(x)

    if k_bias is None:
        k_bias = torch.zeros_like(q_bias)
    qkv_bias = torch.cat((q_bias, k_bias, v_bias))
    if getattr(module, "qkv_bias_separate", False):
        return module.qkv(x) + qkv_bias
    return F.linear(x, module.qkv.weight, qkv_bias)


def normalize_qk(module, q, k):
    q_norm = getattr(module, "q_norm", None)
    k_norm = getattr(module, "k_norm", None)
    return (q_norm(q) if q_norm else q), (k_norm(k) if k_norm else k)


def apply_rope(module, q, k, v, prefix, rope):
    if rope is None:
        return q, k
    if apply_rot_embed_cat is None:
        raise ImportError("timm is required for rotary-position ViT attention.")

    rotate_half = getattr(module, "rotate_half", False)
    q = torch.cat((q[:, :, :prefix], apply_rot_embed_cat(q[:, :, prefix:], rope, half=rotate_half)), dim=2)
    k = torch.cat((k[:, :, :prefix], apply_rot_embed_cat(k[:, :, prefix:], rope, half=rotate_half)), dim=2)
    return q.type_as(v), k.type_as(v)
