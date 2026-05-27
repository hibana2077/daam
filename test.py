from daam import TimmViTDAAM, load_image, prepare_image, overlay_heatmap

# image = load_image("InputImage/ILSVRC2012_val_00000269.JPEG")
image = load_image("InputImage/ILSVRC2012_val_00012653.JPEG")

# with TimmViTDAAM.from_name("vit_tiny_patch16_224", pretrained=True) as daam:
with TimmViTDAAM.from_name("deit_tiny_patch16_224", pretrained=True) as daam:
    result = daam(prepare_image(image, daam.model))

overlay_heatmap(image, result.final_map).save("daam_overlay.png")
print("predicted:", result.predicted_index)
print("saved: daam_overlay.png")