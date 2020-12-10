
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


KAGGLE_PARIS_DATASET = 'data/kaggle_paris'

ARC_DATASET_ROOT = 'data/arc_dataset_raw/arcDataset'

JFR_DATASET = [f'data/jfr_dataset/pour-gregoire{i}' for i in range(1, 4)]
JFR_DATASET_PARIS = 'data/jfr_dataset/raw/paris'
JFR_DATASET_PARIS_FILTERED = 'data/jfr_dataset/raw/paris_filtered'
JFR_DATASET_RELIGION = 'data/jfr_dataset/mosquees-cathedrales'
JFR_DATASET_SELECTION = 'data/jfr_dataset/selection-finale'

GOOGLE_IMAGES = 'data/google_images'

FLICKR_PARIS = 'data/flickr/paris_pictures'


def get_kaggle_paris_images():

    for p in sorted(os.listdir(f'{KAGGLE_PARIS_DATASET}/raw/data_filtered')):
        p_dir = f'{KAGGLE_PARIS_DATASET}/raw/data_filtered/{p}'
        print(p_dir)
        if os.path.isdir(p_dir):
            for e, img in enumerate(sorted([i for i in os.listdir(p_dir) if i!='.DS_Store'])):
                img_ = cv2.imread(f'{p_dir}/{img}')
                if type(img_)==np.ndarray:
                    yield p, e, img_


def get_google_images():

    for p in os.listdir(f'{GOOGLE_IMAGES}/raw'):
        p_dir = f'{GOOGLE_IMAGES}/raw/{p}'
        if os.path.isdir(p_dir):
            for e, img in enumerate(sorted([i for i in os.listdir(p_dir) if i!='.DS_Store'])):
                yield p, e, cv2.imread(f'{p_dir}/{img}'), img


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


def get_jfr_paris_dataset():

    for e, p in enumerate(sorted(os.listdir(JFR_DATASET_PARIS_FILTERED))):
        if p!='.DS_Store':
            try:
                yield p, cv2.imread(f'{JFR_DATASET_PARIS_FILTERED}/{p}')
            except Exception as ex:
                print(ex)


def get_jfr_religion_dataset():

    for path in sorted(os.listdir(JFR_DATASET_RELIGION)):
        if path[0]!='.':
            for e, p in enumerate(sorted(os.listdir(f'{JFR_DATASET_RELIGION}/{path}'))):
                try:
                    yield path, p, cv2.imread(f'{JFR_DATASET_RELIGION}/{path}/{p}')
                except Exception as ex:
                    print(ex)


def get_jfr_selection():

    for p in sorted(os.listdir(JFR_DATASET_SELECTION)):
        yield p, cv2.imread(f'{JFR_DATASET_SELECTION}/{p}')


def get_flickr_paris():

    for img in sorted(os.listdir(f'{FLICKR_PARIS}/pictures')):
        if img!='.DS_Store':
            yield img, cv2.imread(f'{FLICKR_PARIS}/pictures/{img}')


def resize_image(image, resolutions, uniform_background=True):
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
    resized_images = []
    for resolution in resolutions:
        image_resized = cv2.resize(image_t, (resolution, resolution))

        # Set uniform white background (255, 255, 255)
        # It Background may be black or white, we set white everywhere to be consistent
        if uniform_background:
            border = (image_resized==255).all(axis=2) | (image_resized==0).all(axis=2)
            border_rgb = np.repeat(np.expand_dims(border, -1), 3, axis=2)
            image_resized[border_rgb] = 255

        resized_images.append(image_resized)

    return resized_images


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

    from shutil import copyfile
    import matplotlib.pyplot as plt

    # Loop on JFR images
    resolution = 1024
    for e, (name, img) in tqdm(enumerate(get_jfr_dataset()), total=7437):
        print(e, name)
        if not '.psd' in name:
            image, image_flipped = augment_image(resize_image(img, resolution))
            cv2.imwrite(f'data/jfr_dataset_resized_augmented_{resolution}/{e}.jpg', image)
            cv2.imwrite(f'data/jfr_dataset_resized_augmented_{resolution}/{e}_.jpg', image_flipped)

    # JFR Paris images
    resolutions = [256]
    for e, obj in tqdm(enumerate(get_jfr_paris_dataset()), total=2729):
        if obj is not None:
            name, img = obj
            print(e, name)
            if not '.psd' in name:
                for res in resolutions:
                    image, image_flipped = augment_image(resize_image(img, res))
                    cv2.imwrite(f'data/jfr_dataset/jfr_paris_filtered_augmented_{res}/{e}.jpg', image)
                    cv2.imwrite(f'data/jfr_dataset/jfr_paris_filtered_augmented_{res}/{e}_.jpg', image_flipped)

    # JFR religion images
    resolutions = [256, 1024]
    for e, (type_, name, img) in tqdm(enumerate(get_jfr_religion_dataset()), total=492):
        if not '.psd' in name:
            for res in resolutions:
                if e>=306:
                    image, image_flipped = augment_image(resize_image(img, res))
                    cv2.imwrite(f'data/jfr_dataset_religion_resized_augmented_{res}/{type_}_{e}.jpg', image)
                    cv2.imwrite(f'data/jfr_dataset_religion_resized_augmented_{res}/{type_}_{e}_.jpg', image_flipped)

    # JFR selection
    resolutions = [256, 1024]
    for name, img in get_jfr_selection():
        for res in resolutions:
            image, image_flipped = augment_image(resize_image(img, res))
            cv2.imwrite(f'data/jfr_dataset_selection_{res}/{name}', image)
            cv2.imwrite(f'data/jfr_dataset_selection_{res}/flipped/flipped_{name}', image_flipped)

    # Build datasets with all images at both resolutions
    path_names = {
        'jfr_dataset_resized_augmented': 'dataset',
        'jfr_dataset_paris_resized_augmented': 'paris',
        'jfr_dataset_religion_resized_augmented': 'religion'
    }
    for path in ['jfr_dataset_resized_augmented', 'jfr_dataset_paris_resized_augmented', 'jfr_dataset_religion_resized_augmented']:
        for res in [256, 1024]:
            for img in os.listdir(f'data/{path}_{res}'):
                copyfile(f'data/{path}_{res}/{img}', f'data/jfr_dataset_all_{res}/{path_names[path]}_{img}')

    # Build google images
    res = 256
    for name, e, img, _ in get_google_images():
        image, image_flipped = augment_image(resize_image(img, res))
        cv2.imwrite(f'data/google_images/augmented_256/{name}_{e}.jpg', image)
        cv2.imwrite(f'data/google_images/augmented_256/{name}_{e}_.jpg', image_flipped)

    # Kaggle Paris images
    res = 256
    for p, e, img in get_kaggle_paris_images():
        image, image_flipped = augment_image(resize_image(img, res))
        cv2.imwrite(f'{KAGGLE_PARIS_DATASET}/data_filtered_augmented_256/{p}_{e}.jpg', image)
        cv2.imwrite(f'{KAGGLE_PARIS_DATASET}/data_filtered_augmented_256/{p}_{e}_.jpg', image_flipped)

    # FLickr Paris images
    for e, (name, img) in enumerate(get_flickr_paris()):
        if e%100==0:
            print(name)

        image = resize_image(img, res)
        cv2.imwrite(f'{FLICKR_PARIS}/pictures_256/{name}.jpg', image)

    # Resize Flickr images
    for d in ['good']:
        for p in sorted(os.listdir(f'data/flickr/paris_pictures/classification_dataset/{d}')):
            if p!='.DS_Store':
                img = cv2.imread(f'data/flickr/paris_pictures/classification_dataset/{d}/{p}')
                img = resize_image(img, 256)
                cv2.imwrite(f'data/flickr/paris_pictures/classification_dataset/{d}_256/{p}', img)

    # Resize
    root = 'data/flickr/religion_pictures'
    for dir_ in os.listdir(f'{root}/pictures'):
        if dir_!='.DS_Store':
            print(dir_)
            os.makedirs(f'{root}/pictures_256/{dir_}', exist_ok=True)
            for img in os.listdir(f'{root}/pictures/{dir_}'):
                if img !='.DS_Store':
                    img_ = cv2.imread(f'{root}/pictures/{dir_}/{img}')
                    cv2.imwrite(f'{root}/pictures_256/{dir_}/{img}', resize_image(img_, [256], uniform_background=False)[0])

    # Filter google religion images
    root = 'data/google_images/religion'
    for dir_ in os.listdir(f'{root}/raw')[2:]:
        r = f'{root}/raw/{dir_}'
        for i in os.listdir(r):
            if i!='.DS_Store':
                img = cv2.imread(f'{r}/{i}')
                img_256, img_512 = resize_image(img, [256, 512], uniform_background=False)
                cv2.imwrite(f'{root}/pictures_256/{dir_}_{i}', img_256)
                cv2.imwrite(f'{root}/pictures_512/{dir_}_{i}', img_512)

    predictions = pickle.load(open(f'{ROOT}/predictions.pkl', 'rb'))
    plt.plot(sorted(predictions.values())) ; plt.show()

    imgs = set([i for i, j in predictions.items() if j>=0.8])
    for i in os.listdir(f'{root}/pictures_512'):
        if not i in imgs:
            shutil.move(f'{root}/pictures_512/{i}', f'{root}/trash/{i}')

    # Resize JFR religion
    JFR_DATASET_RELIGION = 'data/jfr_dataset/mosquees-cathedrales'
    for path in sorted(os.listdir(JFR_DATASET_RELIGION))[4:]:
        if path[0]!='.' and path!='interieurs':
            print(path)
            for e, p in enumerate(sorted(os.listdir(f'{JFR_DATASET_RELIGION}/{path}'))):
                if p!='.DS_Store':
                    try:
                        img = cv2.imread(f'{JFR_DATASET_RELIGION}/{path}/{p}')
                        img_256, img_512 = resize_image(img, [256, 512], uniform_background=False)
                        cv2.imwrite(f'data/jfr_dataset/jfr_religion_256/{path}_{p}', img_256)
                        cv2.imwrite(f'data/jfr_dataset/jfr_religion_512/{path}_{p}', img_512)
                    except Exception as e:
                        print(e)

    for p in os.listdir('data/jfr_dataset/selection-finale'):
        if p!='.DS_Store' and p!='.BridgeSort':
            img = cv2.imread(f'data/jfr_dataset/selection-finale/{p}')
            img_256, img_512 = resize_image(img, [256, 512], uniform_background=False)
            cv2.imwrite(f'data/jfr_dataset/jfr_religion_256/selection_{p}', img_256)
            cv2.imwrite(f'data/jfr_dataset/jfr_religion_512/selection_{p}', img_512)

