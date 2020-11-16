
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten


TRAIN_DATA_DIR = 'data/classification/train'
TEST_DATA_DIR = 'data/classification/test'

BATCH_SIZE = 16


# Data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='binary')


# Build the model
model = VGG16(include_top=False, input_shape=(256, 256, 3))
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(256, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(class1)

# Define new model
model = Model(inputs=model.inputs, outputs=output)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:19]:
    layer.trainable = False

# Summarize
model.summary()


# Compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=test_generator)




###
# Use classifier
###

from keras.models import load_model
from dataset import resize_image
import cv2
import os
import numpy as np
import shutil


ROOT = 'data/flickr/paris_pictures/pictures'

model = load_model('classification_model.h5')

paths = sorted(os.listdir(ROOT))[4140:]
batch_size = 16

for b in range(0, len(paths), batch_size):
    print(b)
    batch_paths = paths[b: (b + batch_size)]

    # Resize images
    batch_images = []
    for p in batch_paths:
        batch_images.append(np.expand_dims(resize_image(cv2.imread(f'{ROOT}/{p}'), 256) / 255., 0))

    # Predict labels
    predictions = model.predict(np.concatenate(batch_images))

    # Copy files
    for p, pred in zip(batch_paths, predictions):
        c = 1 if pred[0]>=0.5 else 0
        shutil.move(f'{ROOT}/{p}', f'data/flickr/paris_pictures/classified/{c}/{p}')
