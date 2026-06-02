from pathlib import Path

from daam import TimmViTDAAM, load_image, overlay_heatmap, prepare_image


INPUT_DIR = Path("InputImage")
MODEL_NAME = "vit_base_patch16_dinov3"
OUTPUT_PREFIX = "daam_last_layer_overlay"


def iter_input_images(input_dir):
    """Yield input image paths in a stable filename order."""
    return sorted(path for path in input_dir.iterdir() if path.is_file())


def save_last_layer_overlay(daam, image_path):
    """Explain one image and save the final-block DAAM overlay."""
    image = load_image(image_path)
    result = daam(prepare_image(image, daam.model))
    output_path = Path(f"{OUTPUT_PREFIX}_{image_path.name}")

    overlay_heatmap(image, result.last_layer_map).save(output_path)
    print("predicted:", result.predicted_index)
    print(f"saved: {output_path}")


def main():
    """Generate last-layer DAAM overlays for all images in `INPUT_DIR`."""
    with TimmViTDAAM.from_name(MODEL_NAME, pretrained=True) as daam:
        for image_path in iter_input_images(INPUT_DIR):
            save_last_layer_overlay(daam, image_path)


if __name__ == "__main__":
    main()
