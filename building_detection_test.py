
from dataset import resize_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm


MARGIN = 10


DATA_ROOT = 'data/flickr/paris_pictures/0_100'

os.makedirs(f'{DATA_ROOT}/buildings', exist_ok=True)
os.makedirs(f'{DATA_ROOT}/no_buildings', exist_ok=True)

predictions = pickle.load(open(f'{DATA_ROOT}/predictions.pkl', 'rb'))

for image in tqdm(sorted([i for i in os.listdir(f'{DATA_ROOT}/original') if i!='.DS_Store'])):
    # Read original image
    img = np.array(Image.open(f'{DATA_ROOT}/original/{image}'))
    if len(img.shape)==3 and img.shape[2]==3:
        img_256, img_1024 = resize_image(img, [256, 1024], uniform_background=False)

        if f'{image}.jpg' in predictions:

            pred = predictions[f'{image}.jpg']
            if len(pred['scores'])>0 and pred['scores'][0]>0.8:
                # Get a rectangle bosk with the mask
                mask = pred['masks'][:,:,0]
                where = np.array(np.where(mask))
                x1, y1 = np.amin(where, axis=1) 
                x2, y2 = np.amax(where, axis=1)

                x1, y1 = max(0, x1 - MARGIN), max(0, y1 - MARGIN)
                x2, y2 = min(255, x2 + MARGIN), min(255, y2 + MARGIN)

                img_building = img_1024[4*x1 : 4*x2, 4*y1 : 4*y2]
                img_building_256 = resize_image(img_building, [256], uniform_background=False)[0]
                Image.fromarray(img_building_256).save(f'{DATA_ROOT}/buildings/{image}')

            else:
                Image.fromarray(img_256).save(f'{DATA_ROOT}/no_buildings/{image}')

        else:
            Image.fromarray(img_256).save(f'{DATA_ROOT}/no_buildings/{image}')
    else:
        print('Bad shape for ', image)
