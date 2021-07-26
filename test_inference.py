import pixellib
from pixellib.instance import custom_segmentation

segment = custom_segmentation()
segment.inferConfig(num_classes= 1, class_names= ["BG","grano"], detection_nms_threshold = 0.2, detection_threshold = 0.4,
    detection_max_instances = 1000, image_max_dim = 640, image_min_dim = 512, learning_rate = 0.001)
segment.load_model("/home/ubuntu/git/PixelLib/pixellib/mask_rcnn_model.003-2.183297.h5")
segment.segmentImage("/home/ubuntu/git/PixelLib/pixellib/gc_16_90.jpg", show_bboxes=False,
    output_image_name="2lm.png")