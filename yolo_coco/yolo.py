import os
import cv2
import numpy as np

import config

# load the COCO class labels for YOLO model
labels_path = os.path.join(config.cnn_yolo_dir, "coco.names")
labels = open(labels_path, "r").read().strip().split("\n")

# assign random colours to all COCO class labels
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype = "uint8")

def yolo_predict(frame, net, *args):
    '''
    Run the prediction to detect objects in a frame using YOLOv4.

    Parameters
    ----------
    frame : ndarray
        Video frame from which the objects are to be detected and tracked.
    net : 
        YOLOV4 model.
    *args : String
        Class labels that are of interest to the user

    Returns
    -------
    boxes : List
        The bounding box rectangles (tlwh format) of the detected objects.
    confidences : List
        Confidence score of the detected objects.
    classLabels : List
        Class labels of the detected objects

    '''
    
    # initialize lists to append the bounding boxes, confidences and classLabels
    boxes = []
    confidences = []
    classLabels = []
    
    (h,w) = frame.shape[:2]

    # construct a model from the net (YOLOv4) and set input params
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(scale = 1 / 255.0, size = (416, 416), swapRB = True)
    
    # forward pass 
    classIds, scores, bboxes = model.detect(frame, 
                                           confThreshold=config.yolo_thres_confidence)

    # loop over the output to extract desired classes
    for (classId, score, box) in zip(classIds, scores, bboxes):
        if labels[classId] in args:
            # scale the bounding box parameters
            box = box[0:4] * np.array([w, h, w, h])
            (minX, minY, width, height) = box.astype("int")
        
            # find the corner points for cv2.rectangle
            startX = int(minX)
            startY = int(minY)
            
            boxes.append([startX, startY, int(width), int(height)])
            confidences.append(float(score))
            classLabels.append(labels[classId])

    return boxes, confidences, classLabels