from side import *

modelurl = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
vi_vrl ="sample/pexels-mario-angel-5915075.mp4"
classfie = "coco.names"
Detector = detector()
Detector.readClass(classfie)
Detector.model_dowload(modelurl)
Detector.video_detection(vi_vrl)
