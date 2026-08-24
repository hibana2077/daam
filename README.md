# DAAM for timm Vision Transformers

![DAAM attention flow](https://raw.githubusercontent.com/hibana2077/daam/refs/heads/main/Cover.png)

A compact, `timm`-native implementation of **Dynamic Accumulated Attention Map**
(Liao, Gao, Zhang — Pattern Recognition 2025, [arXiv:2503.14640](https://arxiv.org/abs/2503.14640)).
It reveals the *attention flow*: one map per transformer block, accumulated from
the first block to the last, showing how the decision-making token forms its
attention.

```bash
pip install daam-timm-vit
```

## Quick start

```python
import timm
from PIL import Image
from daam import DAAM, heatmap, overlay

model = timm.create_model("deit_small_patch16_224", pretrained=True)
transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))

image = Image.open("InputImage/ILSVRC2012_val_00046384.JPEG").convert("RGB")
with DAAM(model) as daam:
    result = daam(transform(image).unsqueeze(0))          # explains the top-1 class
    # result = daam(x, target=74)                          # or any class index

maps = heatmap(result.maps[0], size=image.size[::-1])     # (num_blocks, H, W) uint8
overlay(image, maps[-1]).save("daam.jpg")                 # final block
for b, hm in enumerate(maps, 1):                          # the whole flow
    overlay(image, hm).save(f"daam_block{b}.jpg")
```

`python example.py [model_name]` runs this on `InputImage/` and writes to `results/`.

## Two variants

The paper defines the channel importance coefficients C_b = ∂Y/∂T_b for two settings.
Pick one with `mode=`, or leave it `None` to infer from the model (`'feature'` iff
`model.num_classes == 0`):

| `mode` | Y | `target` |
| --- | --- | --- |
| `'class'` (needs a head) | class score (Eq. 8) | class index, default top-1 |
| `'feature'` (head ignored; uses the pre-logits feature) | feature · dimension-wise weight w (Eq. 13) | optional `(K, D)` reference features (k-NN memory bank neighbours, class prototypes, …); default: the image's own feature, w_d = f̂_d² |

```python
model = timm.create_model("vit_small_patch14_dinov2", pretrained=True)   # no head → 'feature'
with DAAM(model) as daam:
    result = daam(x)                      # label-free
    result = daam(x, target=bank[idx])    # weight from k-NN neighbours, as in the paper

model = timm.create_model("deit_small_patch16_224", pretrained=True)      # has a head
with DAAM(model) as daam:
    result = daam(x, mode="feature")      # explain the backbone feature, ignore the head
    result = daam(x, mode="class", target=74)
```

## What you get

- `result.block_maps` — `(B, L, h, w)` per-block maps L_b on the patch grid (Eq. 9).
- `result.maps` — `(B, L, h, w)` accumulated maps L_{b,DAAM} (Eq. 11–12).
- `result.output`, `result.target` — logits/features and the explained class.
- `heatmap(maps, size)` — official rendering: scale by the final block's max, `sigmoid(5x) − 0.5`, min-max, bilinear upsample → uint8.
- `overlay(image, hm, alpha=0.4)` — jet-style blend as a PIL image.

## Supported models

Any timm model with `model.blocks[*].attn` exposing `attn_drop`/`proj` (ViT, DeiT,
DINOv2/v3, EVA/EVA02, BEiT, FlexiViT, …) and `global_pool` of `'token'` (the
paper's [CLS] decomposition) or `'avg'` (the mean patch token is decomposed the
same way). Register tokens are handled. The attention matrix is read from timm's
own non-fused path, so RoPE, q/k-norm and relative position bias need no special
casing; `fused_attn` is restored on `close()`.

Custom weights: use timm — `timm.create_model(name, checkpoint_path="my.pt", num_classes=...)`.

## Development

`python test_daam.py` runs the self-check (random weights, CPU). Releases follow the
[PyPI release guide](https://github.com/hibana2077/daam/blob/main/docs/pypi_release_guide.md).

## Citation

```bibtex
@article{yiliaoPR2025dynamic,
  title={Dynamic Accumulated Attention Map for Interpreting Evolution of Decision-making in Vision Transformer},
  author={Liao, Yi and Gao, Yongsheng and Zhang, Weichuan},
  journal={Pattern Recognition},
  volume={165},
  pages={111607},
  year={2025},
  publisher={Elsevier}
}
```

MIT license. Official code: <https://github.com/ly9802/DynamicAccumulatedAttentionMap>
