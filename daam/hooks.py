import math

import torch
from torch.nn import functional as F

try:
    from timm.layers import apply_rot_embed_cat
except Exception:  # timm is imported lazily by users of create_daam.
    apply_rot_embed_cat = None


def prefix_tokens(model):
    """Return how many leading tokens should be skipped before patch tokens."""
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    return 1 if hasattr(model, "cls_token") else 0


def uses_class_token(model):
    """Return whether attribution should focus on the class token."""
    global_pool = getattr(model, "global_pool", "")
    return prefix_tokens(model) > 0 and global_pool in ("", "token", None)


def reshape_tokens(tensor, prefix=1, grid_size=None):
    """Reshape flat ViT tokens into a spatial BCHW patch grid."""
    tokens = tensor[:, prefix:, :] if prefix else tensor
    if grid_size is None:
        side = int(math.sqrt(tokens.shape[1]))
        grid_size = (side, side)
    if grid_size[0] * grid_size[1] != tokens.shape[1]:
        raise ValueError("Could not reshape ViT patch tokens into a spatial grid.")
    return tokens.reshape(tokens.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2)


def collect_vit_layers(model):
    """Collect supported attention modules and their gradient hook layers."""
    if not hasattr(model, "blocks"):
        raise ValueError("Only timm ViT-style models with `model.blocks` are supported.")

    layers = []
    for index, block in enumerate(model.blocks):
        attn = getattr(block, "attn", None)
        proj = getattr(attn, "proj", None) if attn is not None else None
        has_qkv = has_supported_qkv(attn)
        if attn is None or proj is None or not has_qkv:
            raise ValueError(f"Block {index} is not a supported timm ViT attention block.")
        layers.append((attn, select_gradient_capture_layer(attn, proj)))
    return layers


def select_gradient_capture_layer(attn, proj):
    """Pick the layer whose input gradient matches the saved activation shape."""
    norm = getattr(attn, "norm", None)
    if hasattr(norm, "register_forward_hook"):
        return norm
    return proj


def has_supported_qkv(attn):
    """Return whether the attention module exposes a supported QKV projection."""
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
    """Collect DAAM activations and matching attention-output gradients."""

    def __init__(self, model):
        self.model = model
        self.prefix = prefix_tokens(model)
        self.use_class_token = uses_class_token(model)
        self.activations = []
        self.gradients = []
        self.handles = []

        for attn, grad_layer in collect_vit_layers(model):
            self.handles.append(self._forward_hook(attn, self._save_activation))
            self.handles.append(self._forward_hook(grad_layer, self._save_gradient))

    def clear(self):
        """Drop tensors collected during the previous forward/backward pass."""
        self.activations = []
        self.gradients = []

    def close(self):
        """Remove all registered torch hook handles."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x):
        """Run the model after clearing tensors from the previous call."""
        self.clear()
        return self.model(x)

    def _forward_hook(self, module, fn):
        """Register a forward hook that supports old and new torch versions."""
        try:
            return module.register_forward_hook(fn, with_kwargs=True)
        except TypeError:
            return module.register_forward_hook(lambda m, i, o: fn(m, i, {}, o))

    def _save_activation(self, module, inputs, kwargs, output):
        """Save attention-weighted value activations for one ViT block."""
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
            is_causal=kwargs.get("is_causal", False),
            shared_rel_pos=kwargs.get("shared_rel_pos"),
            shared_rel_pos_bias=kwargs.get("shared_rel_pos_bias"),
        )
        self.activations.append(activation.detach().cpu())

    def _save_gradient(self, module, inputs, kwargs, output):
        """Register a backward hook on the tensor entering the gradient layer."""
        if not inputs or not torch.is_tensor(inputs[0]) or not inputs[0].requires_grad:
            return
        inputs[0].register_hook(lambda grad: self._store_gradient(grad))

    def _store_gradient(self, grad):
        """Store the channel weights used to score each saved activation map."""
        if self.use_class_token:
            weight = grad[:, 0, :]
        elif self.prefix:
            weight = grad[:, self.prefix :, :].mean(dim=1)
        else:
            weight = grad.mean(dim=1)
        self.gradients.insert(0, weight.detach().cpu())


def attention_value_product(
    module,
    x,
    prefix=1,
    use_class_token=True,
    rope=None,
    attn_mask=None,
    is_causal=False,
    shared_rel_pos=None,
    shared_rel_pos_bias=None,
):
    """Return attention-weighted values in the same token layout as `x`."""
    q, k, v = project_qkv(module, x)
    q, k = normalize_qk(module, q, k)
    q, k = apply_rope(module, q, k, v, getattr(module, "num_prefix_tokens", prefix), rope)

    attn = (q * module.scale) @ k.transpose(-2, -1)
    attn = apply_attention_bias(
        module,
        attn,
        attn_mask=attn_mask,
        is_causal=is_causal,
        shared_rel_pos=shared_rel_pos,
        shared_rel_pos_bias=shared_rel_pos_bias,
    )
    attn = attn.softmax(dim=-1)

    drop = getattr(module, "attn_drop", None)
    if drop is not None:
        attn = drop(attn)

    focus = attn[:, :, 0] if use_class_token else attn[:, :, prefix:, :].mean(dim=2)
    values = focus.unsqueeze(-1) * v
    return values.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)


def apply_attention_bias(
    module,
    attn,
    attn_mask=None,
    is_causal=False,
    shared_rel_pos=None,
    shared_rel_pos_bias=None,
):
    """Apply mask and relative-position bias terms to attention logits."""
    mask_bias = resolve_attention_mask(
        attn.shape[-1],
        attn,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    if mask_bias is not None:
        attn = attn + mask_bias

    rel_pos = getattr(module, "rel_pos", None)
    if rel_pos is not None:
        try:
            return rel_pos(attn, shared_rel_pos=shared_rel_pos)
        except TypeError:
            if hasattr(rel_pos, "get_bias"):
                attn = attn + as_attention_bias(rel_pos.get_bias(), attn)
            elif callable(rel_pos):
                attn = rel_pos(attn)
            if shared_rel_pos is not None:
                attn = attn + as_attention_bias(shared_rel_pos, attn)
            return attn

    get_rel_pos_bias = getattr(module, "_get_rel_pos_bias", None)
    has_rel_pos_table = getattr(module, "relative_position_bias_table", None) is not None
    if callable(get_rel_pos_bias) and has_rel_pos_table:
        attn = attn + as_attention_bias(get_rel_pos_bias(), attn)
    if shared_rel_pos is not None:
        attn = attn + as_attention_bias(shared_rel_pos, attn)
    if shared_rel_pos_bias is not None:
        attn = attn + as_attention_bias(shared_rel_pos_bias, attn)
    return attn


def resolve_attention_mask(seq_len, attn, attn_mask=None, is_causal=False):
    """Convert causal or user-provided attention masks into additive bias."""
    if is_causal:
        return attn.new_full((seq_len, seq_len), float("-inf")).triu_(1)
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        mask = attn_mask.to(device=attn.device)
        bias = torch.zeros(mask.shape, dtype=attn.dtype, device=attn.device)
        return bias.masked_fill_(~mask, float("-inf"))
    return as_attention_bias(attn_mask, attn)


def as_attention_bias(bias, attn):
    """Move an additive attention bias to the same device and dtype as logits."""
    if bias.device != attn.device or bias.dtype != attn.dtype:
        return bias.to(device=attn.device, dtype=attn.dtype)
    return bias


def project_qkv(module, x):
    """Project input tokens into query, key, and value tensors."""
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
    """Run a timm QKV projection while supporting separate Q/K/V bias tensors."""
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
    """Apply optional query and key normalization layers from timm attention."""
    q_norm = getattr(module, "q_norm", None)
    k_norm = getattr(module, "k_norm", None)
    return (q_norm(q) if q_norm else q), (k_norm(k) if k_norm else k)


def apply_rope(module, q, k, v, prefix, rope):
    """Apply rotary embeddings to patch tokens while preserving prefix tokens."""
    if rope is None:
        return q, k
    if apply_rot_embed_cat is None:
        raise ImportError("timm is required for rotary-position ViT attention.")

    rotate_half = getattr(module, "rotate_half", False)
    rotated_q = apply_rot_embed_cat(q[:, :, prefix:], rope, half=rotate_half)
    rotated_k = apply_rot_embed_cat(k[:, :, prefix:], rope, half=rotate_half)
    q = torch.cat((q[:, :, :prefix], rotated_q), dim=2)
    k = torch.cat((k[:, :, :prefix], rotated_k), dim=2)
    return q.type_as(v), k.type_as(v)
