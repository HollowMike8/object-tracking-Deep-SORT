import cv2

def non_max_suppression(boxes, confidences, yolo_thres_confidence, iou_thres):
    '''
    Suppress overlapping detections.

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    confidences : ndarray
        Detector confidence scores.
    yolo_thres_confidence : float
        Minimum confidence score to filter weak detections.
    iou_thres : float
        IOU threshold used for non-maximum suppression.

    Returns
    -------
    Returns indices of detections that have survived non-max suppression.

    '''
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, yolo_thres_confidence, iou_thres)
    idxs = idxs.flatten()
    
    return idxs