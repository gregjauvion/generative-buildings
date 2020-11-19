
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN



# Comes from here
# https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

DATA_ROOT = 'data/building_detection_dataset'
RESOLUTION = 256

# Build boxes for each image
#image_boxes = {}
#for f in [f'{DATA_ROOT}/annotation_207.json', f'{DATA_ROOT}/annotation_002310.json']:
#    f = json.load(open(f, 'r'))
#    for img, v in f['_via_img_metadata'].items():
#        boxes = v['regions']
#        if len(boxes)==1:
#            attr = boxes[0]['shape_attributes']
#            b = [attr['x'], attr['y'], attr['x'] + attr['width'], attr['y'] + attr['height']]
#            image_boxes[img[:-5]] = [i / 255. for i in b]

#json.dump(image_boxes, open(f'{DATA_ROOT}/image_boxes.json', 'w'))



class BuildingDataset(Dataset):

    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # Read annotations and get the list of images to include in the model
        annotations = json.load(open(f'{DATA_ROOT}/image_boxes.json'))
        images = list(sorted(annotations.keys()))
        train_images, test_images = images[:int(0.8 * len(images))], images[int(0.8 * len(images)):]

        # define one class
        self.add_class("dataset", 1, "building")

        # Add images
        dataset_images = train_images if is_train else test_images
        for image in dataset_images:
            image_path = f'{DATA_ROOT}/data/256/{image}'
            image_id = image
            box = annotations[image]

            # add to dataset
            self.add_image('dataset', image_id=image_id, path=image_path, box=box)

        # Add image with no boxes
        bad_images = os.listdir(f'{DATA_ROOT}/data/256_bad')
        bad_train_images, bad_test_images = bad_images[:int(0.8 * len(bad_images))], bad_images[int(0.8 * len(bad_images)):]
        bad_dataset_images = bad_train_images if is_train else bad_test_images
        for image in bad_dataset_images:
            image_path = f'{DATA_ROOT}/data/256_bad/{image}'
            image_id = image

            # add to dataset
            self.add_image('dataset', image_id=image_id, path=image_path, box=None)
 
    # load the masks for an image
    def load_mask(self, image_id):

        # Get details of image
        info = self.image_info[image_id]
        if info['box'] is None:
            masks = np.zeros([RESOLUTION, RESOLUTION, 0], dtype='uint8')
        else:
            box = (np.clip(np.array(info['box']), 0, 1) * 255).astype(np.uint8)
            masks = np.zeros([RESOLUTION, RESOLUTION, 1], dtype='uint8')
            # create masks
            row_1, row_2 = box[1], box[3]
            col_1, col_2 = box[0], box[2]
            masks[row_1:row_2, col_1:col_2, 0] = 1

        nb_masks = masks.shape[2]

        return masks, np.asarray([self.class_names.index('building')] * nb_masks, dtype='int32')
 

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
 

# train set
train_set = BuildingDataset()
train_set.load_dataset('building', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 
# test/val set
test_set = BuildingDataset()
test_set.load_dataset('building', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# Show dataset
#image, mask = test_set.load_image(10), test_set.load_mask(10)
#plt.imshow(image) ; plt.imshow(mask[0][:,:,0], alpha=0.2) ; plt.show()



# define a configuration for the model
class BuildingConfig(Config):
    # Give the configuration a recognizable name
    NAME = "building_cfg"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 125


config = BuildingConfig()

model = MaskRCNN(mode='training', model_dir='building_detection/', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
