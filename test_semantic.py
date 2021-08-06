import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_image.segmentAsAde20k("/home/ubuntu/Downloads/1.png", output_image_name= "1_seg.png")
