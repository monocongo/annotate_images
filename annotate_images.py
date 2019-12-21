import argparse
import os
from typing import List

import cv2
import imutils
import lxml.etree as etree
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from object_detection.utils import label_map_util


# ------------------------------------------------------------------------------
class BoundingBox:

    def __init__(
            self,
            label: str,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            confidence: float,
    ):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence


# ------------------------------------------------------------------------------
def to_pascal(
        bboxes: List[BoundingBox],
        image_file_name: str,
        img_width: int,
        img_height: int,
        pascal_dir: str,
) -> str:
    """
    Writes a PASCAL VOC (XML) annotation file containing the bounding boxes for
    an image.

    :param bboxes: iterable of bounding box objects
    :param image_file_name: the image file name
    :param pascal_dir: directory where the PASCAL file should be written
    :return: path to the PASCAL VOC file
    """

    # get the image dimensions
    annotation = etree.Element('annotation')
    filename = etree.SubElement(annotation, "filename")
    filename.text = image_file_name
    source = etree.SubElement(annotation, "source")
    database = etree.SubElement(source, "database")
    database.text = "OpenImages"
    size = etree.SubElement(annotation, "size")
    width = etree.SubElement(size, "width")
    width.text = str(img_width)
    height = etree.SubElement(size, "height")
    height.text = str(img_height)
    depth = etree.SubElement(size, "depth")
    depth.text = str(3)
    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = "0"
    for bbox in bboxes:
        obj = etree.SubElement(annotation, "object")
        name = etree.SubElement(obj, "name")
        name.text = bbox.label
        pose = etree.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = etree.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = etree.SubElement(obj, "difficult")
        difficult.text = "0"
        bndbox = etree.SubElement(obj, "bndbox")
        xmin = etree.SubElement(bndbox, "xmin")
        xmin.text = str(bbox.xmin)
        xmax = etree.SubElement(bndbox, "xmax")
        xmax.text = str(bbox.xmax)
        ymin = etree.SubElement(bndbox, "ymin")
        ymin.text = str(bbox.ymin)
        ymax = etree.SubElement(bndbox, "ymax")
        ymax.text = str(bbox.ymax)

    # write the XML to file
    pascal_file_path = os.path.join(pascal_dir, os.path.splitext(image_file_name)[0] + ".xml")
    with open(pascal_file_path, 'w') as pascal_file:
        pascal_file.write(etree.tostring(annotation, pretty_print=True, encoding='utf-8').decode("utf-8"))

    return pascal_file_path


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # Usage:
    #
    # $ python annotate_images.py --model /faster_rcnn/frozen_inference_graph.pb \
    #       --labels /git/models/research/object_detection/data/mscoco_label_map.pbtxt \
    #       --images_dir /data/imgs/fedex/images \
    #       --annotations_dir /data/imgs/fedex/kitti \

    # parse the command line arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="base path for frozen checkpoint detection graph",
    )
    args_parser.add_argument(
        "-l",
        "--labels",
        required=True,
        help="labels file",
    )
    args_parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="Directory containing one or more images to be used as input",
    )
    args_parser.add_argument(
        "--annotations_dir",
        required=True,
        type=str,
        help="Directory where annotations will be written",
    )
    args_parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum confidence probability required for detections",
    )
    args_parser.add_argument(
        "--format",
        required=True,
        type=str,
        choices=["darknet", "kitti", "pascal"],
        help="Annotation format to be used",
    )
    args = vars(args_parser.parse_args())

    # allow for growth if on GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # initialize the model
    model = tf.Graph()

    # create a context manager that makes this model the default one for execution
    with model.as_default():
        # initialize the graph definition
        graphDef = tf.GraphDef()

        # load the graph from disk
        with tf.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")

    # load the class labels from disk
    label_map = label_map_util.load_labelmap(args["labels"])
    total_labels = len(label_map_util.get_label_map_dict(label_map).keys())
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=total_labels,
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # create a colors for each label
    colors = np.random.uniform(0, 255, size=(total_labels, 3))

    # create a session to perform inference
    with model.as_default():
        with tf.Session(graph=model) as sess:
            # grab a reference to the input image tensor and the boxes tensor
            image_tensor = model.get_tensor_by_name("image_tensor:0")
            boxes_tensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scores_tensor = model.get_tensor_by_name("detection_scores:0")
            classes_tensor = model.get_tensor_by_name("detection_classes:0")
            num_detections_tensor = model.get_tensor_by_name("num_detections:0")

            # loop over all files in the directory
            for image_file_name in tqdm(os.listdir(args["images_dir"])):

                # only process JPG files
                if not image_file_name.endswith(".jpg"):
                    continue

                # load the image from disk
                image = cv2.imread(os.path.join(args["images_dir"], image_file_name))
                (height, width) = image.shape[:2]

                # check to see if we should resize along the width
                if (width > height) and (width > 1000):
                    image = imutils.resize(image, width=1000)

                # otherwise, check to see if we should resize along the height
                elif (height > width) and (height > 1000):
                    image = imutils.resize(image, height=1000)

                # prepare the image for detection
                (height, width) = image.shape[:2]
                output = image.copy()
                image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                image = np.expand_dims(image, axis=0)

                # perform inference and compute the bounding boxes,
                # probabilities, and class labels
                (boxes, scores, labels, N) = \
                    sess.run(
                        [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                        feed_dict={image_tensor: image},
                    )

                # squeeze the lists into a single dimension
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                labels = np.squeeze(labels)

                # loop over the bounding box predictions
                bboxes = []
                for (box, score, label) in zip(boxes, scores, labels):

                    # if the predicted probability is less than the minimum
                    # confidence, ignore it
                    if (score < args["confidence"]) or (label not in category_index.keys()):
                        continue

                    # scale the bounding box from the range [0, 1] to [W, H]
                    (start_y, start_x, end_y, end_x) = box
                    start_x = int(start_x * width)
                    start_y = int(start_y * height)
                    end_x = int(end_x * width)
                    end_y = int(end_y * height)

                    bbox = BoundingBox(
                        label=category_index[label]["name"],
                        xmin=start_x,
                        xmax=end_x,
                        ymin=start_y,
                        ymax=end_y,
                        confidence=score,
                    )
                    bboxes.append(bbox)

                # write the bounding boxes into a PASCAL VOC annotation file
                if args["format"] == "pascal":
                    to_pascal(bboxes, image_file_name, width, height, args["annotations_dir"])
                else:
                    raise ValueError(f"Unsupported annotation format: {args['format']}")
