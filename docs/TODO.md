# DAAM Roadmap

## 1. Package API for timm ViT models

- [x] Refactor the original DAAM code into small reusable modules.
- [x] Provide a simple API for timm ViT classifiers.
- [x] Keep the implementation ViT-only and validate unsupported models early.
- [x] Add image preprocessing and heatmap overlay helpers.
- [x] Keep individual Python scripts under 200 lines.

## 2. Streamlit inference UI

- [ ] Build an interactive Streamlit app for inference only.
- [ ] Let users upload/select an image.
- [ ] Let users switch timm ViT model names.
- [ ] Show predicted class, DAAM heatmap, and image overlay.

## 3. Hugging Face Space

- [ ] Add Space-ready files.
- [ ] Test the Streamlit app locally.
- [ ] Push the Space to `hibana2077`.
