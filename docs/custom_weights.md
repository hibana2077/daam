# Custom Weights and Core API

DAAM can explain custom-trained ViT classifiers as long as the runtime model is
still a compatible `timm` ViT object. In other words, the architecture should be
created by `timm.create_model()`, then your checkpoint can replace the weights.

## Load a Custom Checkpoint

```python
from daam import TimmViTDAAM, load_image, overlay_heatmap, prepare_image

image = load_image("InputImage/ILSVRC2012_val_00000269.JPEG")

with TimmViTDAAM.from_name(
    "vit_base_patch16_224",
    pretrained=False,
    checkpoint_path="checkpoints/my_vit.pt",
    num_classes=100,
) as daam:
    tensor = prepare_image(image, daam.model)
    result = daam(tensor)

overlay_heatmap(image, result.final_map).save("custom_daam_overlay.png")
```

Pass the same model arguments used during training, especially `num_classes`,
`img_size`, `in_chans`, and other architecture-changing options. If those do not
match, PyTorch will report shape mismatches during `load_state_dict()`.

## Supported Checkpoint Layouts

`checkpoint_path` accepts either a raw state dict or a checkpoint dictionary with
one of these common keys:

- `state_dict`
- `model`
- `model_state_dict`
- `model_ema`
- `net`
- `network`
- `module`

For custom layouts, pass `checkpoint_key`:

```python
with TimmViTDAAM.from_name(
    "vit_base_patch16_224",
    pretrained=False,
    checkpoint_path="checkpoints/my_vit.pt",
    checkpoint_key="weights",
) as daam:
    ...
```

If the checkpoint was saved from `DistributedDataParallel` or a training wrapper,
common prefixes such as `module.` and `model.` are stripped automatically.

## Already Loaded Models

You can load weights yourself and pass the model object directly:

```python
import timm
import torch
from daam import TimmViTDAAM

model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
checkpoint = torch.load("checkpoints/my_vit.pt", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

with TimmViTDAAM(model) as daam:
    ...
```

This path is useful when your training code needs custom checkpoint conversion
before loading.

## Core Objects

- `TimmViTDAAM`: Main explainer. Use `from_name()` for `timm` model names, or
  pass an already-created model to the constructor.
- `DAAMResult`: Return object containing logits, class indices, cumulative maps,
  and per-block maps.
- `prepare_image()`: Builds the `timm` inference transform and returns a
  `[1, C, H, W]` tensor.
- `overlay_heatmap()`: Blends `DAAMResult.final_map` or
  `DAAMResult.last_layer_map` onto the input image.

