import pixellib
from pixellib.instance import custom_segmentation

segment = custom_segmentation()
segment.inferConfig(num_classes= 1, class_names= ["BG","grano"], detection_nms_threshold = 0.4, detection_threshold = 0.3, detection_max_instances = 600)
segment.load_model("/home/ubuntu/git/PixelLib/pixellib/mask_rcnn_model.003-2.183297.h5")
segment.segmentImage("/home/ubuntu/git/PixelLib/pixellib/gc_16_90.jpg", show_bboxes=True, output_image_name="1.png")