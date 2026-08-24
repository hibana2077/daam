"""python example.py [timm_model_name]  -> results/<model>/<image>_block{b}.jpg + final overlay."""

import sys
from pathlib import Path

import timm
from PIL import Image

from daam import DAAM, heatmap, overlay

name = sys.argv[1] if len(sys.argv) > 1 else "deit_small_patch16_224"
model = timm.create_model(name, pretrained=True)
transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
out_dir = Path("results") / name
out_dir.mkdir(parents=True, exist_ok=True)

with DAAM(model) as daam:
    for path in sorted(Path("InputImage").glob("*.JPEG")):
        image = Image.open(path).convert("RGB")
        result = daam(transform(image).unsqueeze(0))
        maps = heatmap(result.maps[0], size=image.size[::-1])  # (L, H, W) attention flow
        for b, hm in enumerate(maps, 1):
            overlay(image, hm).save(out_dir / f"{path.stem}_block{b}.jpg")
        label = result.target.item() if result.target is not None else "feature"
        print(f"{path.name}: target={label} -> {out_dir}/{path.stem}_block*.jpg")
