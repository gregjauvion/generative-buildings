
from dataset import resize_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm




p1 = pickle.load(open(f'{DATA_ROOT}/predictions_300k_400k.pkl', 'rb'))
p2 = pickle.load(open(f'{DATA_ROOT}/predictions_400k_500k.pkl', 'rb'))

predictions = {i: j for i, j in p1.items()}
for i, j in p2.items():
    predictions[i] = j

for i, j in p3.items():
    predictions[i] = j





MARGIN = 10

DATA_ROOT = 'data/flickr/paris_pictures'

images = sorted([i for i in os.listdir(f'{DATA_ROOT}/0_500k_final') if i!='.DS_Store'])
images_set = set(images)

#predictions = pickle.load(open(f'{DATA_ROOT}/predictions_300k_500k.pkl', 'rb'))
resolutions = [256, 512, 1024]
resolution_max = 2048
factor = int(resolution_max / 256) # 256 is the resolution of the prediction model

for image in tqdm([i for i in images if i[0] in ['4']]):
    if image in images_set:
        # Read original image
        img = np.array(Image.open(f'{DATA_ROOT}/300k_500k/300k_500k_original/{image}'))
        if len(img.shape)==3 and img.shape[2]==3:

            # Resize image at different resolutions
            image_resized = resize_image(img, resolutions + [resolution_max], uniform_background=False)
            image_resolutions = image_resized[:-1]
            image_resolution_max = image_resized[-1]

            # Perform building detection and output cropped images
            pred = predictions[f'{image}.jpg'] if f'{image}.jpg' in predictions else predictions[image]
            if len(pred['scores'])>0 and pred['scores'][0]>0.8:
                # Get a rectangle bosk with the mask
                mask = pred['masks'][:,:,0]
                where = np.array(np.where(mask))
                x1, y1 = np.amin(where, axis=1) 
                x2, y2 = np.amax(where, axis=1)

                x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                x2, y2 = min(256 - 1, x2 + MARGIN), min(256 - 1, y2 + MARGIN)

                img_building = image_resolution_max[factor*x1 : factor*x2, factor*y1 : factor*y2]

                # Resize the image at the wanted resolutions
                img_building_resized = resize_image(img_building, resolutions, uniform_background=False)
                for img, res in zip(img_building_resized, resolutions):
                    Image.fromarray(img).save(f'{DATA_ROOT}/300k_500k/300k_500k_buildings_{res}/{image}')

            else:
                Image.fromarray(img_building).save(f'{DATA_ROOT}/no_buildings/{image}')

        else:
            print('Bad shape for ', image)
    else:
        print('Image not in set.')
