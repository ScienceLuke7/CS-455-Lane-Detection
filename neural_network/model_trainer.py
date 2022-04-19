import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Conv2DTranspose, Input

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
""" gt = json_gt[0]
raw_file = './_datasets/train_set/' + gt['raw_file']
train_image = cv2.imread(raw_file)
train_image = train_image.reshape(-1,720,1280,3)
train_label = cv2.imread('train_labels/test_label_0.jpg')
train_label = train_label.reshape(-1,720,1280,3) """


i = 0
all_images = []
all_labels = []
for _ in json_gt:
    gt = json_gt[i]
    raw_file = './_datasets/train_set/' + gt['raw_file']

    train_image = cv2.imread(raw_file)
    all_images.append(train_image)
    train_label = cv2.imread('train_labels/test_label_{0}.jpg'.format(i))
    all_labels.append(train_label)
    i += 1
all_images: np.ndarray = np.array(all_images)
all_images = all_images.reshape(-1,720,1280,3)
all_labels: np.ndarray = np.array(all_labels)
all_labels = all_labels.reshape(-1,720,1280,3)


tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():

    model = Sequential()
    # defines encoder --------------------------------------------------
    # next 3 lines define the down sampler and may not need it
    model.add(tf.keras.layers.Input(shape=(720,1280,3)))
    model.add(Conv2D(13, kernel_size=(3,3), strides=2, use_bias=True, padding='same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())

    model.add(Conv2D(48, kernel_size=(3,3), strides=2, use_bias=True, padding='same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())

    # next 8 lines define the next 5 layers of non-bottlenecking layers
    for _ in range(5):
        model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(64, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(Conv2D(64, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(BatchNormalization())
        model.add(Dropout(0.03))

    # another down sampling layer and again may not need it
    """ model.add(Conv2D(64, kernel_size=(3,3), strides=2, use_bias=True, padding='same'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization()) """

    # non bottlenecking layers again
    for _ in range(2):
        # dilation rate of 2
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(2, 1)))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # dilation rate of 4
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(4, 1)))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # dilation rate of 8
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(8, 1)))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 8)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        # dilation rate of 16
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(16, 1)))
        model.add(Conv2D(128, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 16)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    # encoder part complete --------------------------------------------


    # defines decoder --------------------------------------------------
    # upsampling layer
    model.add(Conv2DTranspose(64, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    # 2 non-bottle necking layers
    for _ in range(2):
        model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(64, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(Conv2D(64, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(BatchNormalization())
        model.add(Dropout(0))

    # one more upsampling layer
    model.add(Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())

    # 2 non-bottle necking layers
    for _ in range(2):
        model.add(Conv2D(16, kernel_size=(3, 1), strides=1, padding='same'))
        model.add(Conv2D(16, kernel_size=(1, 3), strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(16, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(Conv2D(16, kernel_size=(1, 3), strides=1, padding='same', dilation_rate=(1, 1)))
        model.add(BatchNormalization())
        model.add(Dropout(0))

    #output layer - first argument is 5 because of 5 lanes max
    model.add(Conv2DTranspose(16, kernel_size=(2, 2), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, kernel_size=(2, 2), strides=2, padding='same'))
            
    """ model.add(Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, kernel_size=(2, 2), strides=2, padding='same')) """
    print(model.summary())

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(all_images, all_labels, epochs=500)

model.save("saved_cnn_model")

""" test_image = cv2.imread('_datasets/clips/0313-2/20/10.jpg')
# shows image generated from model outputs
gen_img = model(test_image, training=False).numpy()[0]
# model.predict(test_image)

cv2.imshow('image', gen_img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """


""" print('\n\n')
print(model.metrics_names)
print(model.evaluate(test_image, test_label, batch_size=1)) """

