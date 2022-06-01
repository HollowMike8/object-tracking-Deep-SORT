import os
import sys
import cv2
import time
import imutils
import numpy as np
import matplotlib.pyplot as plt

import config

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from yolo_coco.yolo import yolo_predict

from tools import generate_detections as gdet

#=============================================================================
path_dir: str = r"/content/object-tracking-Deep-SORT"

#=============================================================================
# load input video (race.mp4), intitialize the writer, tracker 
vs = cv2.VideoCapture(os.path.join(config.input_dir, "race.mp4"))

writer = None

# initiate totalFrames processed
totalFrames = 0

# initialize the dictionaries to capture detections and trackings
dets_dict = {}
trks_dict = {}

#=============================================================================
# load the yolov3 model
weightsPath = os.path.join(path_dir, "yolo_coco", "yolov3.weights")
configPath = os.path.join(path_dir, "yolo_coco", "yolov3.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#=============================================================================
max_cosine_distance = 0.5
nn_budget = None
IoU_threshold = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', 
                                                   max_cosine_distance, nn_budget)
tracker = Tracker(metric)

#=============================================================================
# loop over thr frames in the input video
while True:
    (grab, frame) = vs.read()
    
    # to break out of loop at the end of video
    if grab == False:
        break
    
    # convert from BGR to RGB and resize
    frame = imutils.resize(frame, width=600)

    # writing the video
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        writer = cv2.VideoWriter(os.path.join(config.output_dir, "race_DeepSORT.avi"), 
                                 fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    # predict the detections in the current frame
    boxes, confidences, classlabels = yolo_predict(frame, net, "person")

    # apply non-max supression
    idxs = preprocessing.non_max_suppression(boxes, confidences, 
                                             config.yolo_thres_confidence, 
                                             IoU_threshold)
    
    # grab the label names from class IDs
    names = np.array(classlabels)

    # grab the appearance features of the detections  
    features = encoder(frame, boxes)
    
    # initialize the detections
    detections = [Detection(bbox, score, class_name, feature) 
                  for bbox, score, class_name, feature in 
                  zip(boxes, confidences, names, features)]
    
    detections = [detections[i] for i in idxs]

    # initialize the tracker
    tracker.predict()
    tracker.update(detections)

    # available colors for drawing bounding boxes
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
      
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
    
        # write the label name and draw bounding boxes
        cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), 
                      (int(bbox[2]),int(bbox[3])), color, 2)

    # write the sketched frame     
    if writer is not None:
        writer.write(frame)

    # update the totalFrames processed
    totalFrames += 1

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# do a bit of cleanup
vs.release()
#=============================================================================