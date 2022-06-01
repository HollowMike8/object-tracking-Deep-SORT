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
    Run the prediction to detect objects in a frame using YOLOv3.

    Parameters
    ----------
    frame : ndarray
        Video frame from which the objects are to be detected and tracked.
    net : 
        YOLOV3 model.
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
    
    # determine only *output* layer names we need from yolo (3 output layers)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, 
                                 crop=False)
    
    # forward pass 
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    # loop over each layer of the outputs (3)
    for output in layerOutputs:
        # loop over the detections in each output
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # consider only predictions with confidence > threshold
            if confidence > config.yolo_thres_confidence and labels[classID] in args:
                # scale the bounding box parameters
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
    
                # find the corner points for cv2.rectangle
                startX = int(centerX - (width/2))
                startY = int(centerY - (height/2))
                
                boxes.append([startX, startY, int(width), int(height)])
                confidences.append(float(confidence))
                classLabels.append(labels[classID])

    return boxes, confidences, classLabels