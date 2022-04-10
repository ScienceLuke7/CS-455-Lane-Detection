import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
# read each line of json file
json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
i = 0
for _ in json_gt:
    gt = json_gt[i]
    gt_lanes = gt['lanes']
    y_samples = gt['h_samples']
    raw_file = './_datasets/train_set/' + gt['raw_file']

    img = cv2.imread(raw_file)

    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
    img_with_lanes = img.copy()

    for lane in gt_lanes_vis:
        cv2.polylines(img_with_lanes, np.int32([lane]), isClosed=False,
                    color=(0,255,0), thickness=3)

    mask_img = np.zeros_like(img)
    colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]

    for j in range(len(gt_lanes_vis)):
        cv2.polylines(mask_img, np.int32([gt_lanes_vis[j]]), isClosed=False,color=colors[j], thickness=5)

    # create grey-scale label image
    label = np.zeros((720,1280,3), dtype = np.uint8)
    for k in range(len(colors)):
        label[np.where((mask_img == colors[k]).all(axis=2))] = 255
    cv2.imwrite('train_labels/test_label_{0}.jpg'.format(i), img_with_lanes)
    i += 1

""" cv2.imshow('image', mask_img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """

# -----------------------------------------------------
