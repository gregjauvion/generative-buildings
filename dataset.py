
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


ROOT = 'data/raw/arcDataset'


def get_images_paths():
    """
    Get all paths to the images
    """

    images_path = []
    for p in os.listdir(ROOT):
        p_dir = f'{ROOT}/{p}'
        if os.path.isdir(p_dir):
            for img in os.listdir(f'{p_dir}'):
                images_path.append(f'{p_dir}/{img}')

    return images_path


RESOLUTION = 256


def resize_image(image):
    """
    Resize an image to the wanted resolution.
    Same factor is applied along x and y axis
    """

    h, w, _ = image.shape

    # Transform into a square image
    if h>w:
        nb_before, nb_after = int((h-w)/2), h - w - int((h-w)/2)
        before, after = np.zeros((h, nb_before, 3)), np.zeros((h, nb_after, 3))
        image_t = np.concatenate((before, image, after), axis=1)
    elif h<w:
        nb_before, nb_after = int((w-h)/2), w - h - int((w-h)/2)
        before, after = np.zeros((nb_before, w, 3)), np.zeros((nb_after, w, 3))
        image_t = np.concatenate((before, image, after), axis=0)
    elif h==w:
        image_t = image

    return cv2.resize(image_t, (RESOLUTION, RESOLUTION))


def augment_image(image, plot=False):
    """
    Return augmented versions of an image.
    For the moment we only flip the image horizontally
    """

    # Build horizontal flip
    image_flipped = cv2.flip(image, 1)

    if plot:
        fig = plt.figure(figsize=(12, 8))
        g = fig.add_subplot(121) ; plt.imshow(image)
        g = fig.add_subplot(122) ; plt.imshow(image_flipped)
        plt.show()

    return image, image_flipped



if __name__=='__main__':

    # Loop on all images and resize them
    images_paths = get_images_paths()
    for e, img_path in tqdm(enumerate(images_paths), total=len(images_paths)):
        image = cv2.imread(img_path, 1)
        images = augment_image(resize_image(image))

        for ee, img in enumerate(images):
            cv2.imwrite(f'data/resized/{e}_{ee}.jpg', img)


