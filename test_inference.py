import pixellib
from pixellib.instance import custom_segmentation

segment = custom_segmentation()
segment.inferConfig(num_classes= 1, class_names= ["BG","grano"], detection_nms_threshold = 0.3, detection_threshold = 0.3,
    detection_max_instances = 1000, image_max_dim = 1280, image_min_dim = 1024)
segment.load_model("/home/ubuntu/git/PixelLib/pixellib/mask_rcnn_model.007-1.302534-29-07-2021.h5")
segment.segmentImage("/home/ubuntu/git/PixelLib/pixellib/8.png", show_bboxes=False,
                output_image_name="8_6.png")