import pixellib
from pixellib.instance import custom_segmentation

segment = custom_segmentation()
segment.inferConfig(num_classes= 1, class_names= ["BG","grano"])
segment.load_model("/home/ubuntu/git/PixelLib/pixellib/mask_rcnn_model.003-2.183297.h5")
segment.segmentImage("/home/ubuntu/git/PixelLib/pixellib/10.png", show_bboxes=True, output_image_name="10_inf_02.png")