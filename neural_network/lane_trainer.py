import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
# read each line of json file
json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
gt = json_gt[0]
gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = './_datasets/train_set/' + gt['raw_file']
# see the image
img = cv2.imread(raw_file)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()