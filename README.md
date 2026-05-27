# DAAM for timm ViT

Reusable DAAM visualization helpers for timm Vision Transformer classifiers.

```python
from daam import TimmViTDAAM, load_image, overlay_heatmap, prepare_image

image = load_image("InputImage/ILSVRC2012_val_00000269.JPEG")

with TimmViTDAAM.from_name("vit_base_patch16_224", pretrained=True) as daam:
    tensor = prepare_image(image, daam.model)
    result = daam(tensor)

overlay = overlay_heatmap(image, result.final_map)
overlay.save("daam_overlay.png")
```

The first package target is intentionally ViT-only. Models must be timm
classifiers with `model.blocks[*].attn` attention blocks.
