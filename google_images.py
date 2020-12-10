
import os
import cv2
from dataset import resize_image
from tqdm import tqdm

ROOT = 'data/google_images/paris'

def list_dir(dir):
    return sorted([i for i in os.listdir(dir) if i!='.DS_Store'])


for dir_ in list_dir(f'{ROOT}/raw'):
    print(dir_)
    for img in tqdm(list_dir(f'{ROOT}/raw/{dir_}')):
        source, nb = img.split('.')[0].split('_')[0], img.split('.')[0].split('_')[1]
        img_ = cv2.imread(f'{ROOT}/raw/{dir_}/{img}')

        img_256, img_512 = resize_image(img_, [256, 512], uniform_background=False)
        cv2.imwrite(f'{ROOT}/paris_256/{dir_}_{source}_{nb}.jpg', img_256)
        cv2.imwrite(f'{ROOT}/paris_512/{dir_}_{source}_{nb}.jpg', img_512)




# Kaggle
ROOT_KAGGLE = 'data/kaggle_paris'
for i in ['paris', 'paris2']:
    for dir_ in list_dir(f'{ROOT_KAGGLE}/{i}'):
        for img in tqdm(list_dir(f'{ROOT_KAGGLE}/{i}/{dir_}')):
            img_ = cv2.imread(f'{ROOT_KAGGLE}/{i}/{dir_}/{img}')
            if img_ is not None:
                img_256, img_512 = resize_image(img_, [256, 512], uniform_background=False)
                cv2.imwrite(f'{ROOT_KAGGLE}/kaggle_paris_256/{dir_}_{img}', img_256)
                cv2.imwrite(f'{ROOT_KAGGLE}/kaggle_paris_512/{dir_}_{img}', img_512)



# Read classifier predictions and classify images
import matplotlib.pyplot as plt
import pickle
import shutil

ROOT = 'data/kaggle_paris'

predictions = pickle.load(open(f'{ROOT}/predictions_kaggle.pkl', 'rb'))

plt.plot(sorted(predictions.values())) ; plt.show()

for img, p in predictions.items():
    c = 1 if p>=0.5 else 0
    shutil.copy(f'{ROOT}/kaggle_paris_256/{img}', f'{ROOT}/classified/{c}/{img}')

pickle.dump([i for i, j in predictions.items() if j>=0.5], open(f'{ROOT}/file_names.pkl', 'wb'))


###
# Resize
###

import cv2
from dataset import resize_image
import os

ROOT = 'data/google_images/religion_selection'

for i in os.listdir(f'{ROOT}/download_images_bing'):
    if i!='.DS_Store':
        for j in os.listdir(f'{ROOT}/download_images_bing/{i}'):
            if j!='.DS_Store':
                img = cv2.imread(f'{ROOT}/download_images_bing/{i}/{j}')
                if img is not None:
                    img_256, img_512 = resize_image(img, [256, 512])
                    cv2.imwrite(f'{ROOT}/pictures_256/{i}_{j}', img_256)
                    cv2.imwrite(f'{ROOT}/pictures_512/{i}_{j}', img_512)


# Delete some pictures
preds = pickle.load(open(f'{ROOT}/predictions.pkl', 'rb'))

imgs = set([i for i, j in preds.items() if j>=0.8])

nb = 0
for img in os.listdir(f'{ROOT}/pictures_512'):
    if not img in imgs:
        shutil.move(f'{ROOT}/pictures_512/{img}', f'{ROOT}/trash/{img}')
