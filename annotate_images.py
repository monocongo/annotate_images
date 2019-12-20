import argparse
import os

import cv2
import imutils
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util


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
        default=0.5,
        help="Minimum confidence probability required for detections",
    )
    args = vars(args_parser.parse_args())

    # initialize a set of colors for our class labels

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
    categoryIdx = label_map_util.create_category_index(categories)

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
            for file_name in os.listdir(args["images_dir"]):

                # only process JPG files
                if not file_name.endswith(".jpg"):
                    continue

                # load the image from disk
                image = cv2.imread(os.path.join(args["images_dir"], file_name))
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
                for (box, score, label) in zip(boxes, scores, labels):

                    # if the predicted probability is less than the minimum
                    # confidence, ignore it
                    if score < args["confidence"]:
                        continue

                    # scale the bounding box from the range [0, 1] to [W, H]
                    (start_y, start_x, end_y, end_x) = box
                    start_x = int(start_x * width)
                    start_y = int(start_y * height)
                    end_x = int(end_x * width)
                    end_y = int(end_y * height)

                    # draw the prediction on the output image
                    label = categoryIdx[label]
                    idx = int(label["id"]) - 1
                    label = "{}: {:.2f}".format(label["name"], score)
                    cv2.rectangle(output, (start_x, start_y), (end_x, end_y),
                                  colors[idx], 2)
                    y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                    cv2.putText(output, label, (start_x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[idx], 1)

                # show the output image
                cv2.imshow("Output", output)
                cv2.waitKey(0)
