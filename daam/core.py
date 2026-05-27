from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .hooks import ViTAttentionHooks, collect_vit_layers, prefix_tokens, reshape_tokens


@dataclass
class DAAMResult:
    logits: torch.Tensor
    target_index: int
    predicted_index: int
    maps: list[np.ndarray]

    @property
    def final_map(self):
        return self.maps[-1] if self.maps else None

    @property
    def probabilities(self):
        return torch.softmax(self.logits, dim=-1)


class TimmViTDAAM:
    """DAAM explainer for timm ViT image classifiers."""

    def __init__(self, model, device=None, normalize_blocks=True):
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.normalize_blocks = normalize_blocks
        self.prefix = prefix_tokens(model)
        collect_vit_layers(model)
        self.hooks = ViTAttentionHooks(self.model)

    @classmethod
    def from_name(cls, model_name, pretrained=True, device=None, normalize_blocks=True, **model_kwargs):
        import timm

        model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)
        return cls(model, device=device, normalize_blocks=normalize_blocks)

    def close(self):
        self.hooks.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __call__(self, image_tensor, target_index=None):
        return self.explain(image_tensor, target_index=target_index)

    def explain(self, image_tensor, target_index=None):
        if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
            raise ValueError("DAAM currently expects one image tensor shaped [1, C, H, W].")

        image_tensor = image_tensor.to(self.device).requires_grad_(True)
        target_size = tuple(image_tensor.shape[-2:])

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self._logits(self.hooks(image_tensor))
            predicted = int(logits.argmax(dim=-1).item())
            target = predicted if target_index is None else int(target_index)
            logits[:, target].sum().backward()

        maps = self._build_maps(target_size)
        return DAAMResult(
            logits=logits.detach().cpu(),
            target_index=target,
            predicted_index=predicted,
            maps=maps,
        )

    def _logits(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("The timm model must return a tensor of class logits.")
        return output

    def _build_maps(self, target_size):
        if len(self.hooks.activations) != len(self.hooks.gradients):
            raise RuntimeError("Could not collect matching DAAM activations and gradients.")

        block_maps = []
        for activation, gradient in zip(self.hooks.activations, self.hooks.gradients):
            token_count = activation.shape[1] - self.prefix
            activation = reshape_tokens(activation, prefix=self.prefix, grid_size=infer_grid(self.model, target_size, token_count))
            weight = gradient[:, :, None, None].clamp(min=0)
            block_map = (activation * weight).sum(dim=1, keepdim=True).clamp(min=0)
            if self.normalize_blocks:
                block_map = normalize_tensor(block_map)
            block_maps.append(block_map)

        patch_size = common_patch_size(block_maps)
        block_maps = [resize_tensor(cam, patch_size) for cam in block_maps]
        total = sum(block_maps)
        final_scale = float(total.max().clamp(min=1e-10))

        maps = []
        running = torch.zeros_like(block_maps[0])
        for block_map in block_maps:
            running = running + block_map
            cam = torch.sigmoid(5.0 * running / final_scale) - 0.5
            cam = normalize_tensor(cam)
            cam = resize_tensor(cam, target_size)[0, 0]
            maps.append((cam.numpy() * 255).astype(np.uint8))
        return maps


def create_daam(model_name="vit_base_patch16_224", pretrained=True, device=None, **model_kwargs):
    return TimmViTDAAM.from_name(model_name, pretrained=pretrained, device=device, **model_kwargs)


def list_timm_vit_models(pretrained=False):
    import timm

    names = timm.list_models(pretrained=pretrained)
    markers = ("vit", "deit", "beit", "eva", "flexivit", "dinov2")
    blocked = ("swin", "maxvit", "tiny_vit", "levit")
    return [name for name in names if any(m in name for m in markers) and not any(b in name for b in blocked)]


def pick_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_tensor(tensor):
    flat = tensor.flatten(1)
    low = flat.min(dim=1).values[:, None, None, None]
    high = flat.max(dim=1).values[:, None, None, None]
    return (tensor - low) / (high - low).clamp(min=1e-10)


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False).cpu()


def common_patch_size(maps):
    height = max(cam.shape[-2] for cam in maps)
    width = max(cam.shape[-1] for cam in maps)
    return height, width


def infer_grid(model, image_size, token_count):
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
