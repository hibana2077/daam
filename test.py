import os
from daam import TimmViTDAAM, load_image, prepare_image, overlay_heatmap

# image = load_image("InputImage/ILSVRC2012_val_00000269.JPEG")
# image = load_image("InputImage/ILSVRC2012_val_00012653.JPEG")
image_lists = sorted(os.listdir("InputImage/"))

with TimmViTDAAM.from_name("vit_base_patch16_dinov3", pretrained=True) as daam:
    for image_name in image_lists:
        image = load_image(os.path.join("InputImage/", image_name))
        result = daam(prepare_image(image, daam.model))

        overlay_heatmap(image, result.last_layer_map).save(f"daam_last_layer_overlay_{image_name}")
        print("predicted:", result.predicted_index)
        print(f"saved: daam_last_layer_overlay_{image_name}")
