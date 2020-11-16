
import numpy as np
import random
import flickrapi
import urllib
import time
import pickle
from tqdm import tqdm


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



if __name__=='__main__':

    #KEY = 'adaaa5f61c029082e06abaee5084c92f'
    #ECRET = '0304e447b10f4e71'

    KEY = 'ad996b51438408c75ba3df8be977f84f'
    SECRET = '5d1dec91461d93b5'

    photo_ids_lat_lon = get_photo_ids_lat_lon()
    photo_id_urls = pickle.load(open('flickr/photo_id_urls_0_200000.pkl', 'rb'))

    # Flickr API calls to get the files urls
    api = flickrapi.FlickrAPI(KEY, SECRET)
    for e, (photo_id, lat, lon) in tqdm(enumerate(photo_ids_lat_lon)):
        if e>=85280 and e<=200000:
            photo_id_urls[photo_id] = {}
            try:
                photo_id_urls[photo_id] = get_sizes_urls(api, photo_id)
            except:
                print('API error', e)

            # Save the file from time to time
            if e%5000==0:
                pickle.dump(photo_id_urls, open(f'flickr/photo_id_urls_{e}.pkl', 'wb'))

            # Wait a bit to not be kicked from the API
            time.sleep(1)

    # Downlaod the images
    #photo_id_urls = pickle.load(open('paris_pictures/urls/photo_id_urls_50000.pkl', 'rb'))
    #for e in range(50000):
    #    if e>=308:
    #        if e%100==0:
    #            print(e)

    #        photo_id = photo_ids_lat_lon[e][0]
    #        urls = photo_id_urls[photo_id]
    #        if len(urls)>0:
                # Select the size with best resolution
    #            sizes = [(int(i[0]), int(i[1])) for i in urls.keys()]
    #            size = max(sizes, key=lambda i: i[0] * i[1])

    #            try:
    #                urllib.request.urlretrieve(urls[(str(size[0]), str(size[1]))], f'paris_pictures/pictures/{e:06d}_{photo_id}.jpg')
    #            except:
    #                print('Download error', e)

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
