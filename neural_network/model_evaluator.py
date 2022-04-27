import json
import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
gt = json_gt[1]
raw_file = './_datasets/train_set/' + gt['raw_file']

train_image = cv2.imread(raw_file)
train_image = train_image.reshape(-1,720,1280,3)
train_label = cv2.imread('train_labels/test_label_1.jpg')
train_label = train_label.reshape(-1,720,1280,3)

model = keras.models.load_model('complex_model_100')

print(model.summary())

model.evaluate(train_image, train_label)

gen_img = model(train_image, training=False).numpy()[0]
img = cv2.cvtColor(gen_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
cv2.imwrite('resultImage_simple_500.jpg', gen_img)

cv2.imshow('image', gen_img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()