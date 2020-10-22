
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


ARC_DATASET_ROOT = 'data/arc_dataset_raw/arcDataset'

JFR_DATASET = [f'data/jfr_dataset/pour-gregoire{i}' for i in range(1, 4)]


def get_arc_dataset():
    """
    Generator on Arc Dataset images
    """

    for p in os.listdir(ARC_DATASET_ROOT):
        p_dir = f'{ARC_DATASET_ROOT}/{p}'
        if os.path.isdir(p_dir):
            for img in os.listdir(f'{p_dir}'):
                yield cv2.imread(f'{p_dir}/{img}')


def get_jfr_dataset():
    """
    Generator on JFR dataset
    """

    for path in JFR_DATASET:
        for p in sorted(os.listdir(path)):
            if p!='.DS_Store':
                yield p, cv2.imread(f'{path}/{p}')



def resize_image(image, resolution):
    """
    Resize an image to the wanted resolution.
    Same factor is applied along x and y axis
    """

    h, w, _ = image.shape

    # Transform into a square image
    if h>w:
        nb_before, nb_after = int((h-w)/2), h - w - int((h-w)/2)
        before, after = np.zeros((h, nb_before, 3)) + 255, np.zeros((h, nb_after, 3)) + 255
        image_t = np.concatenate((before, image, after), axis=1).astype(np.uint8)
    elif h<w:
        nb_before, nb_after = int((w-h)/2), w - h - int((w-h)/2)
        before, after = np.zeros((nb_before, w, 3)) + 255, np.zeros((nb_after, w, 3)) + 255
        image_t = np.concatenate((before, image, after), axis=0).astype(np.uint8)
    elif h==w:
        image_t = image

    # Resize image
    image_resized = cv2.resize(image_t, (resolution, resolution))

    # Set uniform white background (255, 255, 255)
    # It Background may be black or white, we set white everywhere to be consistent
    border = (image_resized==255).all(axis=2) | (image_resized==0).all(axis=2)
    border_rgb = np.repeat(np.expand_dims(border, -1), 3, axis=2)
    image_resized[border_rgb] = 255

    return image_resized


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

    # Loop on JFR images
    for e, (name, img) in tqdm(enumerate(get_jfr_dataset()), total=7437):
        print(e, name)
        if not '.psd' in name:
            image, image_flipped = augment_image(resize_image(img, 512))
            cv2.imwrite(f'data/jfr_dataset_resized_augmented_512/{e}.jpg', image)
            cv2.imwrite(f'data/jfr_dataset_resized_augmented_512/{e}_.jpg', image_flipped)


    # Loop on all images and resize them
    #images_paths = get_images_paths()
    #for e, img_path in tqdm(enumerate(images_paths), total=len(images_paths)):
    #    image = cv2.imread(img_path, 1)
        
        #images = augment_image(resize_image(image))
        #for ee, img in enumerate(images):
        #    cv2.imwrite(f'data/resized_augmented/{e}_{ee}.jpg', img)

    #    cv2.imwrite(f'data/resized/{e}.jpg', resize_image(image))
