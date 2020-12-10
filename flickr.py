
import numpy as np
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


def get_photo_ids(api, keyword, limit, min_date, max_date):
    """
    Returns all photo ids for a given keyword.
    limit is the number of photo ids we want
    Looks ot be blocked after 5000 pictures
    """

    # Get photo_ids
    photos = api.walk(text=keyword, per_page=1000, sort='relevance', min_taken_date=min_date, max_taken_date=max_date)
    photo_ids = set()
    for e, photo in enumerate(photos):
        if e%1000==0:
            print(e, len(photo_ids))
            time.sleep(1)

        if e>=limit:
            break
        photo_ids.add(photo.attrib['id'])

    return photo_ids


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
        # Select the size with lower resolution but higher than 1E6 pixels
        sizes = sorted([(int(i[0]), int(i[1])) for i in urls.keys() if i!=('','')], key=lambda i: i[0] * i[1])
        # Remove too big pictures
        sizes = [s for s in sizes if s[0]*s[1]<1E7]
        sizes_t = [s for s in sizes if s[0]*s[1]>=1E6]
        if len(sizes_t)==0:
            size = sizes[-1]
        else:
            size = sizes_t[0]

        try:
            urllib.request.urlretrieve(urls[(str(size[0]), str(size[1]))], path)
            return True
        except:
            print('Download error', photo_id)

    return False



if __name__=='__main__':

    from keras.models import load_model
    import cv2
    import os

    #ROOT = 'data/flickr/paris_pictures'
    ROOT = 'data/flickr/religion_pictures'

    keys = [
        ('adaaa5f61c029082e06abaee5084c92f', '0304e447b10f4e71'),
        ('ad996b51438408c75ba3df8be977f84f', '5d1dec91461d93b5'),
        ('999dd5b305c2f859d79702916f435ea1', '7a90da2014c8adb9'),
        ('bd91833b1fac14dea46d865485f6d516', 'c076844e38aea337'),
        ('2166e224ce4e118bd9cfe513108ba3ab', 'ea0a5b5f17ea8de1')
    ]

    # Flickr API calls to get the files urls
    ind = 0
    photo_ids = sorted(pickle.load(open(f'{ROOT}/photo_ids.pkl', 'rb')))
    photo_id_urls = {}
    api = flickrapi.FlickrAPI(keys[ind][0], keys[ind][1])
    for e, photo_id in tqdm(enumerate(photo_ids[200000:])):
        photo_id_urls[photo_id] = {}
        try:
            photo_id_urls[photo_id] = get_sizes_urls(api, photo_id)
        except:
            print('API error', e)

        # Wait a bit to not be kicked from the API
        time.sleep(1)

    pickle.dump(photo_id_urls, open(f'{ROOT}/photo_id_urls_5.pkl', 'wb'))

    # Download images without filtering
    keyword_photoid_urls = pickle.load(open(f'{ROOT}/keyword_photoid_urls.pkl', 'rb'))
    for k, p_u in keyword_photoid_urls.items():
        path = f'{ROOT}/pictures/{k}'
        os.makedirs(path, exist_ok=True)
        for p in sorted(p_u.keys())[:300]:
            download_image(p_u, p, f'{path}/{k}_{p}.jpg')

    # Downlaod the images
    model = load_model(f'{ROOT}/classification_model.h5')
    photo_id_urls = pickle.load(open(f'{ROOT}/photo_id_urls.pkl', 'rb'))
    root = 'data/flickr/paris_pictures/pictures'
    batch_nb, batch_paths, batch_images = 0, [], []
    for e, (photo_id, urls) in tqdm(enumerate(photo_id_urls.items()), total=len(photo_id_urls)):
        if e<=1577:
            continue

        if e%100==0:
            print(e)

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
            predictions = model.predict(np.array(batch_images) / 255.).reshape(-1)
            print(predictions)

            # Delete bad images
            for e_p, p in enumerate(predictions):
                if p<0.8:
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

    # Return Paris photo ids, looping month per month from 2017 to 2020
    api = flickrapi.FlickrAPI(keys[0][0], keys[0][1])
    keywords = ['mosque', 'buddhist temple', 'cathedral', 'church', 'evangelist church', 'orthodox church', 'protestant temple', 'synagogue', 'temple']
    keywords_photoids = {k: set() for k in keywords}
    for year in [2020]:
        for month in range(5, 12):
            print(year, month)
            min_date = f'{year}-{month}-01'
            if month==12:
                max_date = f'{year+1}-01-01'
            else:
                max_date = f'{year}-{month+1}-01'

            print(min_date, max_date)

            for k in keywords:
                ids = get_photo_ids(api, k, 5000, min_date, max_date)
                print(k, len(ids))
                for i in ids:
                    keywords_photoids[k].add(i)

    # Return photo ids for different keywords
    limit = 2000000
    keyword_photoids = {}
    keywords = ['mosque', 'buddhist temple', 'cathedral', 'church', 'evangelist church', 'orthodox church', 'protestant temple', 'synagogue', 'temple']
    for keyword in ['paris']:
        print(keyword)
        keyword_photoids[keyword.replace(' ', '_')] = get_photo_ids(api, keyword, limit)

    # Download URLs
    ROOT = 'data/flickr/religion_pictures'
    keywords = ['mosque', 'buddhist temple', 'cathedral', 'church', 'evangelist church', 'orthodox church', 'protestant temple', 'synagogue', 'temple']
    keyword_photoids = pickle.load(open(f'{ROOT}/keywords_photoids.pkl', 'rb'))
    ind = 8
    key, secret = keys[ind-5]
    photoid_urls = {}
    api = flickrapi.FlickrAPI(key, secret)
    photo_ids = sorted(list(keyword_photoids[keywords[ind]]))[:50000]
    for e, photo_id in enumerate(tqdm(photo_ids)):
        if e%1000==0:
            pickle.dump(photoid_urls, open(f'{ROOT}/{keywords[ind]}.pkl', 'wb'))

        # Wait a bit to not be kicked from the API
        time.sleep(1)

        photoid_urls[photo_id] = {}
        try:
            photoid_urls[photo_id] = get_sizes_urls(api, photo_id)
        except:
            print('API error', photo_id)

    # Download pictures per keyword
    root = 'data/flickr/religion_pictures'
    keywords = ['mosque', 'buddhist_temple', 'cathedral', 'church', 'evangelist_church', 'orthodox_church', 'protestant_temple', 'synagogue', 'temple']
    keyword_photoid_predictions = pickle.load(open(f'{root}/keyword_photoid_predictions.pkl', 'rb'))
    keyword_photoid_urls = pickle.load(open(f'{root}/keyword_photoid_urls.pkl', 'rb'))

    for k in keywords[1:]:
        print(k)
        photo_ids = [i.replace('.jpg', '').split('_')[1] for i, j in keyword_photoid_predictions[k].items() if j>=0.8]
        print(len(photo_ids))

        for e, p in enumerate(tqdm(photo_ids)):
            name = f'{k}_{p}.jpg'
            download_image(keyword_photoid_urls[k], p, f'{root}/tmp/{name}')
            img = cv2.imread(f'{root}/tmp/{name}')
            if img is not None:
                img_256, img_512 = resize_image(img, [256, 512], uniform_background=False)
                cv2.imwrite(f'{root}/pictures_256/{name}', img_256)
                cv2.imwrite(f'{root}/pictures_512/{name}', img_512)
                os.remove(f'{root}/tmp/{name}')
