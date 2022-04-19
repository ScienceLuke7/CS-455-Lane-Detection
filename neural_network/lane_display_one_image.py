import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
# read each line of json file
json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
""" gt = json_gt[0]
gt_lanes = gt['lanes']
y_samples = gt['h_samples']
raw_file = './_datasets/train_set/' + gt['raw_file']
img = cv2.imread(raw_file) """


raw_images = []
all_lanes = []
all_y_samples = []

i = 0
for _ in json_gt:
    curr_gt = json_gt[i]

    curr_gt_lanes = curr_gt['lanes']
    all_lanes.append(curr_gt_lanes)

    curr_y_samples = curr_gt['h_samples']
    all_y_samples.append(curr_y_samples)

    curr_raw_file = './_datasets/train_set/' + curr_gt['raw_file']
    raw_images.append(cv2.imread(curr_raw_file))
    i += 1

# Displays just the image
""" 
cv2.imshow('image', img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """
#------------------------------------------------------


# Displays image with JSON lanes on it
# -----------------------------------------------------
all_gt_lanes = []
for gt_lanes in all_lanes:
    all_gt_lanes.append([[[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for y_samples in all_y_samples] for lane in gt_lanes])

all_img_with_lanes = []
all_img_with_lanes.append(img.copy() for img in raw_images)

i = 0
for gt_lanes_vis in all_gt_lanes:
    for lane in gt_lanes_vis:
        cv2.polylines(all_img_with_lanes[i], np.int32([lane]), isClosed=False,
                    color=(0,255,0), thickness=3)
        i += 1

"""
cv2.imshow('image', img_with_lanes)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """
# -----------------------------------------------------


# Displays masking image with lanes
# -----------------------------------------------------

mask_img = np.zeros_like(img)
colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]

for i in range(len(gt_lanes_vis)):
    cv2.polylines(mask_img, np.int32([gt_lanes_vis[i]]), isClosed=False,color=colors[i], thickness=5)

# create grey-scale label image
label = np.zeros((720,1280,3), dtype = np.uint8)
for i in range(len(colors)):
   label[np.where((mask_img == colors[i]).all(axis=2))] = 255


for img_with_lanes in all_img_with_lanes:
    cv2.imwrite('/labels/test_label_{0}.jpg'.format(), img_with_lanes)

""" cv2.imshow('image', mask_img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """

# -----------------------------------------------------
