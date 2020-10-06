


###########
# Try to remove background from images
###########


image_path = images_path[1]

# Read image as numpy array
image = cv2.imread(image_path, 1)
h, w = image.shape[:2]
original = image.copy()

# Build the mask
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Opening
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# Contours
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for c in cnts:
    cv2.drawContours(mask, [c], -1, (255,255,255), -1)
    break

# Closing
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
close = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(original, original, mask=close)

# Apply the mask
result[close==0] = (255,255,255)

fig = plt.figure(figsize=(12, 8))
g = fig.add_subplot(1, 2, 1)
plt.imshow(original)
g = fig.add_subplot(1, 2, 2)
plt.imshow(original - result)

plt.show()


################
# Another try
################

from skimage import io as skio
from skimage import filters


img = skio.imread(images_path[1000])
sobel = filters.sobel(img)
plt.imshow(sobel)

blurred = filters.gaussian(sobel, sigma=2.0)
plt.imshow(blurred) ; plt.show()



################
# Still another try
################

image_path = images_path[1000]

# Read image as numpy array
image = cv2.imread(image_path, 1)

# Apply Canny edge detector
# Thresholds have been fixed empirically
edges = cv2.Canny(image, 150, 300)

# Detection of crop parameters (top, bottom, left, right)
edges_row, edges_col = edges.sum(axis=1)>0, edges.sum(axis=0)>0
crop_top, crop_bottom = np.argmax(edges_row), len(edges_row) - np.argmax(edges_row[::-1])
crop_left, crop_right = np.argmax(edges_col), len(edges_col) - np.argmax(edges_col[::-1])

# Crop image
image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]


