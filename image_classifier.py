
import numpy as np
import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint


TRAIN_DATA_DIR = 'data/train'
TEST_DATA_DIR = 'data/test'

BATCH_SIZE = 16


# Split into train/test
p_0 = 'data/flickr/religion_pictures/classification_dataset/0'
p_1 = 'data/flickr/religion_pictures/classification_dataset/1'
img_0 = [i for i in os.listdir(p_0) if i!='.DS_Store']
img_1 = [i for i in os.listdir(p_1) if i!='.DS_Store']

train_0 = np.random.choice(img_0, size=int(0.75*len(img_0)), replace=False)
train_1 = np.random.choice(img_1, size=int(0.75*len(img_1)), replace=False)
test_0 = [i for i in img_0 if not i in set(train_0)]
test_1 = [i for i in img_1 if not i in set(train_1)]

for i in train_0:
    shutil.copy(f'{p_0}/{i}', f'data/flickr/religion_pictures/classification_dataset/train/0/{i}')

for i in test_0:
    shutil.copy(f'{p_0}/{i}', f'data/flickr/religion_pictures/classification_dataset/test/0/{i}')

for i in train_1:
    shutil.copy(f'{p_1}/{i}', f'data/flickr/religion_pictures/classification_dataset/train/1/{i}')

for i in test_1:
    shutil.copy(f'{p_1}/{i}', f'data/flickr/religion_pictures/classification_dataset/test/1/{i}')


# Data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
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

#pred = np.concatenate([model.predict(test_generator[i][0]) for i in range(len(test_generator))])
#labels = np.concatenate([test_generator[i][1] for i in range(len(test_generator))]).astype(np.bool)
#np.sum((pred.reshape(-1)>0.5) == labels) / len(pred)

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
    callbacks=ModelCheckpoint('classification_model_3_bis.h5', save_best_only=True),
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


ROOT = 'data'

model = load_model('classification_model_3_bis.h5')

paths = [p for p in sorted(os.listdir(f'{ROOT}/pictures/')) if p!='.DS_Store']
batch_size = 16

for b in range(0, len(paths), batch_size):
    print(b)
    batch_paths = paths[b: (b + batch_size)]

    # Resize images
    batch_images = []
    for p in batch_paths:
        batch_images.append(np.expand_dims(resize_image(cv2.imread(f'{ROOT}/pictures/{p}'), [256])[0] / 255., 0))

    # Predict labels
    predictions = model.predict(np.concatenate(batch_images))
    all_p.extend(list(predictions.reshape(-1)))

    # Copy files
    for p, pred in zip(batch_paths, predictions):
        c = 1 if pred[0]>=0.5 else 0
        shutil.copy(f'{ROOT}/pictures/{p}', f'{ROOT}/classified/{c}/{p}')





###
# Copy all images
###

import pickle
import matplotlib.pyplot as plt
import os
import shutil

ROOT = 'data/flickr/paris_pictures'

for root in [f'{ROOT}/{r}/{r}_buildings_512' for r in ['0_100k', '100k_200k', '200k_300k', '300k_500k']]:
    print(root)
    for f in os.listdir(root):
        shutil.copy(f'{root}/{f}', f'{ROOT}/0_500k_buildings_512/{f}')

