import pixellib
from pixellib.semantic import semantic_segmentation

segment_video = semantic_segmentation()
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_video.process_video_ade20k("/home/ubuntu/Downloads/2.mp4", overlay = True,
                                    frames_per_second= 20, output_video_name="output_video_3.mp4")