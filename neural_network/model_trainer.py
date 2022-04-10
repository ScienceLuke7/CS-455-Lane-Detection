import json
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Conv2DTranspose, Input


# define shape
""" input = Input(shape=(720, 1280, 3))
conv1 = Conv2D(13, kernel_size=(3,3), strides=2, use_bias=True, padding='same', input_shape=(1, 3, 3, 4))(input)
maxpool1 = MaxPool2D(pool_size=(2,2), strides=2)(conv1)
norm1 = BatchNormalization()(maxpool1) """

json_gt = [json.loads(line) for line in open('./_datasets/train_set/label_data_0313.json')]
i = 0

all_images = []
all_labels = []
for _ in json_gt:
    gt = json_gt[i]
    raw_file = './_datasets/train_set/' + gt['raw_file']

    train_image = cv2.imread(raw_file)
    train_image = train_image.reshape(-1,720,1280,3)
    all_images.append(train_image)
    train_label = cv2.imread('train_labels/test_label_{0}.jpg'.format(i))
    train_label = train_label.reshape(-1,720,1280,3)
    all_labels.append(train_label)
    i += 1



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
model.fit(all_images, all_labels, epochs=5000, batch_size=32)

""" gen_img = model(test_image, training=False).numpy()[0]

cv2.imshow('image', gen_img)
cv2.moveWindow('image', 400, 100)
while cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 1:
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows() """

model.save("saved_cnn_model")

""" print('\n\n')
print(model.metrics_names)
print(model.evaluate(test_image, test_label, batch_size=1)) """

