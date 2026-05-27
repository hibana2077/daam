import numpy as np
import torch
from PIL import Image


def load_image(path_or_file):
    """Load an image as a PIL `RGB` image.

    Parameters:
        path_or_file: Filesystem path or file-like object accepted by PIL.
    """
    return Image.open(path_or_file).convert("RGB")


def build_transform(model):
    """Create the inference transform expected by a `timm` model.

    The transform is resolved from `timm.data.resolve_model_data_config(model)`,
    so custom models should keep their data config attached when possible.
    """
    import timm

    config = timm.data.resolve_model_data_config(model)
    return timm.data.create_transform(**config, is_training=False)


def prepare_image(image, model):
    """Convert one image into a model-ready tensor shaped `[1, C, H, W]`.

    Parameters:
        image: PIL image, path, or file-like object.
        model: Model used to resolve the `timm` inference transform.
    """
    if not isinstance(image, Image.Image):
        image = load_image(image)
    return build_transform(model)(image).unsqueeze(0)


def overlay_heatmap(image, heatmap, alpha=0.45):
    """Blend a `uint8` DAAM heatmap on top of a PIL `RGB` image.

    Parameters:
        image: Base image as a PIL image, path, or file-like object.
        heatmap: DAAM heatmap, usually `DAAMResult.final_map`.
        alpha: Blend strength for the colored heatmap.

    Returns:
        A PIL `RGB` image containing the overlay.
    """
    if not isinstance(image, Image.Image):
        image = load_image(image)

    heatmap = Image.fromarray(np.asarray(heatmap, dtype=np.uint8)).resize(image.size)
    heatmap = colorize_heatmap(np.asarray(heatmap))

    base = np.asarray(image).astype(np.float32)
    colored = heatmap.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * colored
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def colorize_heatmap(heatmap):
    """Convert a grayscale heatmap into an RGB heatmap array."""
    heatmap = np.asarray(heatmap, dtype=np.float32) / 255.0
    colors = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    colors[..., 0] = np.clip(1.5 * heatmap, 0, 1)
    colors[..., 1] = np.clip(1.5 * (1.0 - np.abs(heatmap - 0.5) * 2.0), 0, 1)
    colors[..., 2] = np.clip(1.5 * (1.0 - heatmap), 0, 1)
    return (colors * 255).astype(np.uint8)


def topk_labels(logits, k=5, labels=None):
    """Return top-k class predictions from model logits.

    Parameters:
        logits: Model logits shaped `[1, num_classes]`.
        k: Number of predictions to return.
        labels: Optional label list indexed by class id.
    """
    probs = torch.softmax(logits, dim=-1)[0]
    values, indices = probs.topk(k)
    rows = []
    for value, index in zip(values.tolist(), indices.tolist()):
        name = labels[index] if labels and index < len(labels) else str(index)
        rows.append({"index": index, "label": name, "probability": value})
    return rows
