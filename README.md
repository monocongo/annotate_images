# annotate_images
This repository contains a script to perform annotation of images using pre-trained object detection model(s). 

1. Obtain pre-trained model from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). 
For this example we'll use a [Faster-RCNN model with a ResNet50 backbone trained on COCO dataset](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz).
Once we've downloaded the compressed file we'll extract the frozen inference graph 
protobuf file which is the pre-trained model we'll use with our script.
```bash
$ wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz
$ tar -xzf faster_rcnn_nas_coco_2018_01_28.tar.gz faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb
$ export MODEL=`pwd`/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb
```
2. Clone the TensorFlow object detection models API from GitHub:
```bash
$ git clone git@github.com:tensorflow/models.git
$ cd models
$ export TFOD=`pwd`
```
3. Run the script, specifying the model file, the COCO labels prototext file (located 
in the TensorFlow models API), the images directory containing the images, and the 
directory where annotations should be written. 
```bash
$ python annotate_images.py --model ${MODEL} \
> --labels ${TFOD}/research/object_detection/data/mscoco_label_map.pbtxt \
> --images_dir /data/imgs/fedex/images \
> --annotations_dir /data/imgs/fedex/kitti
```
