
import numpy as np
import random
import flickrapi
import urllib
import time
import pickle
from tqdm import tqdm
from dataset import resize_image


def get_sizes_urls(api, photo_id):

    urls = {}

    # API call
    response = api.photos.getSizes(photo_id=photo_id)
    status = response.attrib['stat']
    if status=='ok':
        sizes = response[0]
        for size in sizes:
            attr = size.attrib
            if ('width' in attr) and ('height' in attr) and ('source' in attr):
                urls[(attr['width'], attr['height'])] = attr['source']
    else:
        print('Status not OK', photo_id)

    return urls


def get_photo_ids_lat_lon():
    """
    Read images_flickr_shuffle.txt file
    """

    photo_ids_lat_lon = []
    with open('flickr/images_flickr_shuffle.txt', 'r') as f:
        for l in f.readlines():
            l = l.strip().split()
            photo_ids_lat_lon.append((l[0].split("_")[0], float(l[1]), float(l[2])))

    return photo_ids_lat_lon


def download_image(photo_id_urls, photo_id, path):

    if not photo_id in photo_id_urls:
        print(f'{photo_id} not in dict.')
        return False
        
    urls = photo_id_urls[photo_id]
    if len(urls)>0:
        # Select the size with best resolution
        sizes = [(int(i[0]), int(i[1])) for i in urls.keys()]
        size = max(sizes, key=lambda i: i[0] * i[1])

        try:
            urllib.request.urlretrieve(urls[(str(size[0]), str(size[1]))], path)
            return True
        except:
            print('Download error', e)

    return False



if __name__=='__main__':

    from keras.models import load_model
    import cv2
    import os

    photo_ids_lat_lon = get_photo_ids_lat_lon()
    photo_id_urls = pickle.load(open('flickr/photo_id_urls_300k.pkl', 'rb'))

    # Flickr API calls to get the files urls
    api = flickrapi.FlickrAPI(KEY, SECRET)
    for e, (photo_id, lat, lon) in tqdm(enumerate(photo_ids_lat_lon)):
        if e>=300000 and e<=400000:
            photo_id_urls[photo_id] = {}
            try:
                photo_id_urls[photo_id] = get_sizes_urls(api, photo_id)
            except:
                print('API error', e)

            # Save the file from time to time
            if e%10000==0:
                pickle.dump(photo_id_urls, open(f'flickr/photo_id_urls_1_{e}.pkl', 'wb'))

            # Wait a bit to not be kicked from the API
            time.sleep(1)

    # Downlaod the images
    model = load_model('classification_model.h5')
    root = 'data/flickr/paris_pictures/250_300'
    batch_nb, batch_paths, batch_images = 0, [], []
    for e in range(260000, 300000):
        if e%100==0:
            print(e)

        photo_id = photo_ids_lat_lon[e][0]

        # Build image paths
        f_path = f'{e:06d}_{photo_id}.jpg'
        path_o, path_256 = f'{root}/original/{f_path}', f'{root}/256/{f_path}'
        res = download_image(photo_id_urls, photo_id, path_o)

        if res:
            img_o = cv2.imread(path_o)
            if img_o is not None:
                # Resize
                img_256 = resize_image(img_o, [256], uniform_background=False)[0]
                batch_images.append(img_256)
                cv2.imwrite(path_256, img_256)
                batch_nb += 1
                batch_paths.append((path_o, path_256))

        if batch_nb==16:
            # Prediction
            predictions = model.predict(np.array(batch_images)).reshape(-1)

            # Delete bad images
            for e_p, p in enumerate(predictions):
                if p<0.5:
                    os.remove(batch_paths[e_p][0])
                    os.remove(batch_paths[e_p][1])

            batch_nb, batch_paths, batch_images = 0, [], []

    # Clean the 256 folder
    p_original = 'data/flickr/paris_pictures/tmp'
    p_256 = 'data/flickr/paris_pictures/tmp_256'
    i_original = [i for i in os.listdir(p_original) if i!='.DS_Store']
    i_256 = [i for i in os.listdir(p_256) if i!='.DS_Store']
    to_delete = [i for i in i_256 if not i in set(i_original)]
    for d in to_delete:
        os.remove(f'{p_256}/{d}')



########
# Download images for a gven keyword
########

# Comes from
# https://medium.com/@adrianmrit/creating-simple-image-datasets-with-flickr-api-2f19c164d82f

#keyword = 'mosque'
#api = flickrapi.FlickrAPI(KEY, SECRET)

# Get photo_ids
#photos = api.walk(text=keyword, per_page=50, sort='relevance')
#photo_ids = []
#for e, photo in enumerate(photos):
#    if e<=300:
#        photo_ids.append(photo.attrib['id'])


# Get URLs
#photo_id_urls = {}
#for p in photo_ids[:50]:
#    photo_id_urls[p] = get_sizes_urls(api, p)

# Download pictures
#for photo_id, urls in photo_id_urls.items():
#    if len(urls)>0:
        # Select the size with best resolution
#        sizes = [(int(i[0]), int(i[1])) for i in urls.keys()]
#        size = max(sizes, key=lambda i: i[0] * i[1])

#        urllib.request.urlretrieve(urls[(str(size[0]), str(size[1]))], f'religion_pictures/mosque/{photo_id}.jpg')
