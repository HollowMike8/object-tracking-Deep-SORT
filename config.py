import os

# base path directory
path_dir = "/content/object-tracking-Deep-SORT"

# define the directory path to the input videos
input_dir = os.path.join(path_dir, "data")

# define the directory path to the output videos
output_dir = os.path.join(path_dir, "output")

# define the directory path to the appearance descriptor model
cnn_mars128_dir = os.path.join(path_dir, "model_data")

# define the directory path to the yolo-coco model
cnn_yolo_dir = os.path.join(path_dir, "yolo_coco")

# define the directory path to the deep sort folder
deep_sort_dir = os.path.join(path_dir, "deep_sort")

# define the class label which is of interest
label = "person"

# define threshold confidence to filter weak detections
yolo_thres_confidence = 0.8
