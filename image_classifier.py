
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint


TRAIN_DATA_DIR = 'data/classification_2/train'
TEST_DATA_DIR = 'data/classification_2/test'

BATCH_SIZE = 16


# Split into train/test
p_0 = 'data/flickr/paris_pictures/classification_dataset/bad_256'
p_1 = 'data/flickr/paris_pictures/classification_dataset/good_256'
img_0 = os.listdir(p_0)
img_1 = os.listdir(p_1)

train_0 = np.random.choice(img_0, size=len(img_0) - 1000, replace=False)
train_1 = np.random.choice(img_1, size=len(img_1) - 1000, replace=False)
test_0 = [i for i in img_0 if not i in set(train_0)]
test_1 = [i for i in img_1 if not i in set(train_1)]

for i in train_0:
    shutil.copy(f'{p_0}/{i}', f'data/flickr/paris_pictures/classification_dataset/train/0/{i}')

for i in test_0:
    shutil.copy(f'{p_0}/{i}', f'data/flickr/paris_pictures/classification_dataset/test/0/{i}')

for i in train_1:
    shutil.copy(f'{p_1}/{i}', f'data/flickr/paris_pictures/classification_dataset/train/1/{i}')

for i in test_1:
    shutil.copy(f'{p_1}/{i}', f'data/flickr/paris_pictures/classification_dataset/test/1/{i}')


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
    callbacks=ModelCheckpoint('classification_model_2.h5', save_best_only=True),
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


ROOT = 'data/pictures'

model = load_model('classification_model_after_building.h5')

paths = [p for p in sorted(os.listdir(f'{ROOT}/buildings')) if p!='.DS_Store']
batch_size = 32

for b in range(0, len(paths), batch_size):
    print(b)
    batch_paths = paths[b: (b + batch_size)]

    # Resize images
    batch_images = []
    for p in batch_paths:
        batch_images.append(np.expand_dims(resize_image(cv2.imread(f'{ROOT}/buildings/{p}'), [256])[0] / 255., 0))

    # Predict labels
    predictions = model.predict(np.concatenate(batch_images))

    # Copy files
    for p, pred in zip(batch_paths, predictions):
        c = 1 if pred[0]>=0.9 else 0
        shutil.move(f'{ROOT}/buildings/{p}', f'{ROOT}/{c}/{p}')
