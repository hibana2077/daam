from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .hooks import ViTAttentionHooks, collect_vit_layers, prefix_tokens, reshape_tokens


CHECKPOINT_STATE_KEYS = (
    "state_dict",
    "model",
    "model_state_dict",
    "model_ema",
    "net",
    "network",
    "module",
)
STATE_DICT_PREFIXES = ("module.", "model.")
SUPPORTED_TIMM_PREFIXES = (
    "beit",
    "deit",
    "eva",
    "flexivit",
    "naflexvit",
    "vit_",
    "vitamin",
)
BLOCKED_TIMM_PREFIXES = (
    "convit_",
    "crossvit_",
    "davit_",
    "efficientvit_",
    "fastvit_",
    "gcvit_",
    "gemma4_vit_",
    "levit_",
    "maxvit_",
    "maxxvit_",
    "maxxvitv2_",
    "mobilevit_",
    "mobilevitv2_",
    "mvitv2_",
    "nextvit_",
    "repvit_",
    "samvit_",
    "shvit_",
    "test_vit",
    "tiny_vit_",
)
BLOCKED_TIMM_NAMES = {
    "vit_base_patch16_18x2_224",
    "vit_base_patch16_xp_224",
    "vit_dlittle_patch16_reg1_gap_256",
    "vit_dpwee_patch16_reg1_gap_256",
    "vit_dwee_patch16_reg1_gap_256",
    "vit_huge_patch14_xp_224",
    "vit_large_patch14_xp_224",
    "vit_pwee_patch16_reg1_gap_256",
    "vit_small_patch16_18x2_224",
}


@dataclass
class DAAMResult:
    """Result returned by `TimmViTDAAM.explain()`.

    Attributes:
        logits: Raw class logits returned by the model, moved to CPU.
        target_index: Class index used for the backward attribution pass.
        predicted_index: Top-1 class index from `logits`.
        maps: Cumulative DAAM maps, one per transformer block, as `uint8` arrays.
        layer_maps: Per-block DAAM maps before cumulative accumulation.
    """

    logits: torch.Tensor
    target_index: int
    predicted_index: int
    maps: list[np.ndarray]
    layer_maps: list[np.ndarray] | None = None

    @property
    def final_map(self):
        """Final cumulative DAAM map as a `uint8` image array, or `None`."""
        return self.maps[-1] if self.maps else None

    @property
    def last_layer_map(self):
        """Last transformer block DAAM map as a `uint8` image array, or `None`."""
        return self.layer_maps[-1] if self.layer_maps else None

    @property
    def probabilities(self):
        """Softmax probabilities computed from `logits`."""
        return torch.softmax(self.logits, dim=-1)


class TimmViTDAAM:
    """DAAM explainer for compatible `timm` Vision Transformer classifiers.

    Use this class when you already have a model object, including a `timm`
    model with custom weights loaded manually.

    Parameters:
        model: A `timm` ViT-style classifier with `model.blocks[*].attn`.
        device: Target device. Use `"auto"` or `None` to prefer CUDA when
            available.
        normalize_blocks: Normalize each block map before cumulative rendering.
            Defaults to `False` to match the reference DAAM scripts.
    """

    def __init__(self, model, device=None, normalize_blocks=False):
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.normalize_blocks = normalize_blocks
        self.prefix = prefix_tokens(model)
        collect_vit_layers(model)
        self.hooks = ViTAttentionHooks(self.model)

    @classmethod
    def from_name(
        cls,
        model_name,
        pretrained=True,
        device=None,
        normalize_blocks=False,
        checkpoint_path=None,
        state_dict=None,
        checkpoint_key=None,
        strict=True,
        **model_kwargs,
    ):
        """Build a DAAM explainer from a `timm` model name.

        This keeps the architecture in `timm.create_model()` and optionally
        loads user-provided weights into that same architecture.

        Parameters:
            model_name: Model name accepted by `timm.create_model()`.
            pretrained: Whether to load official `timm` pretrained weights.
            device: Target device. Use `"auto"` or `None` to prefer CUDA.
            normalize_blocks: Normalize each block map before accumulation.
                Defaults to `False` to match the reference DAAM scripts.
            checkpoint_path: Optional path to a PyTorch checkpoint file.
            state_dict: Optional state dict or checkpoint mapping already loaded
                in memory.
            checkpoint_key: Optional key to select a state dict inside a custom
                checkpoint mapping.
            strict: Passed to `model.load_state_dict()`.
            **model_kwargs: Forwarded to `timm.create_model()`, for example
                `num_classes`, `img_size`, or `in_chans`.

        Returns:
            A ready-to-use `TimmViTDAAM` instance.
        """
        import timm

        if checkpoint_path is not None and state_dict is not None:
            raise ValueError("Use either `checkpoint_path` or `state_dict`, not both.")

        model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)
        if checkpoint_path is not None:
            state_dict = load_checkpoint_state_dict(checkpoint_path, checkpoint_key=checkpoint_key)
        elif state_dict is not None:
            state_dict = extract_state_dict(state_dict, checkpoint_key=checkpoint_key)
        if state_dict is not None:
            load_model_state_dict(model, state_dict, strict=strict)
        return cls(model, device=device, normalize_blocks=normalize_blocks)

    def close(self):
        """Remove registered model hooks.

        Call this manually when not using `with TimmViTDAAM(...) as daam:`.
        """
        self.hooks.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __call__(self, image_tensor, target_index=None):
        """Alias for `explain(image_tensor, target_index=target_index)`."""
        return self.explain(image_tensor, target_index=target_index)

    def explain(self, image_tensor, target_index=None):
        """Generate DAAM maps for one image tensor.

        Parameters:
            image_tensor: A normalized tensor shaped `[1, C, H, W]`.
            target_index: Optional class index to explain. When omitted, DAAM
                explains the predicted top-1 class.

        Returns:
            A `DAAMResult` with logits, the explained class, cumulative maps,
            and per-block maps.
        """
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("DAAM currently expects one image tensor shaped [1, C, H, W].")

        image_tensor = image_tensor.to(self.device).detach().requires_grad_(True)
        target_size = tuple(image_tensor.shape[-2:])

        param_states = [param.requires_grad for param in self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad_(False)
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = self._logits(self.hooks(image_tensor))
                predicted, target = self._select_target_index(logits, target_index)
                logits[:, target].sum().backward()

            maps, layer_maps = self._build_maps(target_size)
        finally:
            for param, requires_grad in zip(self.model.parameters(), param_states):
                param.requires_grad_(requires_grad)
        return DAAMResult(
            logits=logits.detach().cpu(),
            target_index=target,
            predicted_index=predicted,
            maps=maps,
            layer_maps=layer_maps,
        )

    def _logits(self, output):
        """Return logits from a model output and reject unsupported outputs."""
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("The timm model must return a tensor of class logits.")
        return output

    def _select_target_index(self, logits, target_index):
        """Choose the explained class and validate it against the logits size."""
        predicted = int(logits.argmax(dim=-1).item())
        target = predicted if target_index is None else int(target_index)
        class_count = logits.shape[-1]
        if target < 0 or target >= class_count:
            raise ValueError(f"target_index must be in [0, {class_count - 1}], got {target}.")
        return predicted, target

    def _build_maps(self, target_size):
        """Build cumulative and per-block DAAM maps from saved hook tensors."""
        if not self.hooks.activations:
            raise RuntimeError("Could not collect DAAM activations from the model.")
        if len(self.hooks.activations) != len(self.hooks.gradients):
            activation_count = len(self.hooks.activations)
            gradient_count = len(self.hooks.gradients)
            raise RuntimeError(
                "Could not collect matching DAAM activations and gradients "
                f"({activation_count} activations, {gradient_count} gradients)."
            )

        block_maps = []
        for activation, gradient in zip(self.hooks.activations, self.hooks.gradients):
            token_count = activation.shape[1] - self.prefix
            activation = reshape_tokens(
                activation.float(),
                prefix=self.prefix,
                grid_size=infer_grid(self.model, target_size, token_count),
            )
            weight = gradient.float()[:, :, None, None].clamp(min=0)
            block_map = (activation * weight).sum(dim=1, keepdim=True).clamp(min=0)
            if self.normalize_blocks:
                block_map = normalize_tensor(block_map)
            block_maps.append(block_map)

        patch_size = common_patch_size(block_maps)
        block_maps = [resize_tensor(cam, patch_size) for cam in block_maps]
        layer_maps = [render_map(block_map, target_size) for block_map in block_maps]
        total = sum(block_maps)
        final_scale = float(total.max().clamp(min=1e-10))

        maps = []
        running = torch.zeros_like(block_maps[0])
        for block_map in block_maps:
            running = running + block_map
            maps.append(render_map(running, target_size, scale=final_scale))
        return maps, layer_maps


def create_daam(
    model_name="vit_base_patch16_224",
    pretrained=True,
    device=None,
    normalize_blocks=False,
    checkpoint_path=None,
    state_dict=None,
    checkpoint_key=None,
    strict=True,
    **model_kwargs,
):
    """Convenience wrapper around `TimmViTDAAM.from_name()`.

    Parameters match `TimmViTDAAM.from_name()`. This is useful for simple scripts
    that prefer a function-style API.
    """
    return TimmViTDAAM.from_name(
        model_name,
        pretrained=pretrained,
        device=device,
        normalize_blocks=normalize_blocks,
        checkpoint_path=checkpoint_path,
        state_dict=state_dict,
        checkpoint_key=checkpoint_key,
        strict=strict,
        **model_kwargs,
    )


def load_checkpoint_state_dict(checkpoint_path, checkpoint_key=None):
    """Load a PyTorch checkpoint and return a model state dict.

    The loader accepts a raw state dict or common checkpoint mappings containing
    keys such as `state_dict`, `model`, `model_state_dict`, or `model_ema`.
    Use `checkpoint_key` for custom checkpoint layouts.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return extract_state_dict(checkpoint, checkpoint_key=checkpoint_key)


def extract_state_dict(checkpoint, checkpoint_key=None):
    """Extract a model state dict from a checkpoint mapping.

    This also strips common wrapper prefixes like `module.` and `model.` so
    checkpoints saved from DDP or training wrappers can be loaded into a plain
    `timm` model.
    """
    if checkpoint_key is not None:
        if not isinstance(checkpoint, Mapping) or checkpoint_key not in checkpoint:
            available = sorted(checkpoint.keys()) if isinstance(checkpoint, Mapping) else []
            raise KeyError(
                f"Checkpoint key {checkpoint_key!r} not found. "
                f"Available keys: {available}"
            )
        checkpoint = checkpoint[checkpoint_key]

    if is_state_dict(checkpoint):
        return strip_state_dict_prefixes(checkpoint)

    if isinstance(checkpoint, Mapping):
        for key in CHECKPOINT_STATE_KEYS:
            value = checkpoint.get(key)
            if is_state_dict(value):
                return strip_state_dict_prefixes(value)
        available = sorted(checkpoint.keys())
        raise ValueError(
            "Could not find a model state_dict in the checkpoint. "
            f"Pass `checkpoint_key` explicitly. Available keys: {available}"
        )

    raise TypeError("Checkpoint must be a state_dict or a mapping that contains one.")


def is_state_dict(value):
    """Return `True` when `value` looks like a PyTorch model state dict."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def strip_state_dict_prefixes(state_dict):
    """Remove wrapper prefixes repeatedly while every key shares the prefix."""
    state_dict = dict(state_dict)
    while state_dict:
        keys = tuple(state_dict.keys())
        prefix = next(
            (item for item in STATE_DICT_PREFIXES if all(key.startswith(item) for key in keys)),
            None,
        )
        if prefix is None:
            return state_dict
        state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def load_model_state_dict(model, state_dict, strict=True):
    """Load a state dict into `model` after stripping common wrapper prefixes."""
    return model.load_state_dict(strip_state_dict_prefixes(state_dict), strict=strict)


def list_timm_vit_models(pretrained=False):
    """List `timm` model names expected to work with this DAAM implementation.

    Parameters:
        pretrained: Forwarded to `timm.list_models(pretrained=...)`.

    Returns:
        Supported ViT-style model names after filtering out known incompatible
        ViT-named families.
    """
    import timm

    names = timm.list_models(pretrained=pretrained)
    return [
        name
        for name in names
        if name.startswith(SUPPORTED_TIMM_PREFIXES)
        and not name.startswith(BLOCKED_TIMM_PREFIXES)
        and name not in BLOCKED_TIMM_NAMES
    ]


def pick_device(device):
    """Resolve a user-provided device string into a concrete torch device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_tensor(tensor):
    """Normalize each batch item independently into the `[0, 1]` range."""
    flat = tensor.flatten(1)
    low = flat.min(dim=1).values[:, None, None, None]
    high = flat.max(dim=1).values[:, None, None, None]
    return (tensor - low) / (high - low).clamp(min=1e-10)


def resize_tensor(tensor, size):
    """Resize a BCHW tensor with bilinear interpolation and move it to CPU."""
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False).cpu()


def render_map(tensor, target_size, scale=None):
    """Render a DAAM tensor as a `uint8` heatmap matching `target_size`."""
    if scale is None:
        scale = float(tensor.max().clamp(min=1e-10))
    cam = torch.sigmoid(5.0 * tensor / scale) - 0.5
    cam = normalize_tensor(cam)
    cam = (cam[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return resize_heatmap_array(cam, target_size)


def resize_heatmap_array(heatmap, target_size):
    """Resize a 2D heatmap array to `(height, width)` when needed."""
    if heatmap.shape == tuple(target_size):
        return heatmap
    height, width = target_size
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(Image.fromarray(heatmap).resize((width, height), resampling))


def common_patch_size(maps):
    """Return the largest spatial size shared by a list of DAAM tensors."""
    height = max(cam.shape[-2] for cam in maps)
    width = max(cam.shape[-1] for cam in maps)
    return height, width


def infer_grid(model, image_size, token_count):
    """Infer the ViT patch grid used to reshape flat patch tokens."""
    patch_embed = getattr(model, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if patch_size:
        grid = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        if grid[0] * grid[1] == token_count:
            return grid

    grid = getattr(patch_embed, "grid_size", None)
    if grid and grid[0] * grid[1] == token_count:
        return tuple(grid)
    return None
