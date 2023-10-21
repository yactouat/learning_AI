import cv2
from PIL import Image
from IPython.display import display

def binarize_img(path, threshold=127, max=255):
    grayscale_img = cv2.imread(path, 0) # 0 = grayscale
    _, binary_img = cv2.threshold(grayscale_img, threshold, max, cv2.THRESH_BINARY)
    return binary_img

def crop_img(img, x, y, width, height):
    return img[y:y+height, x:x+width]

def extract_img_from_bounding_box(img, bounding_box):
    x, y, width, height = bounding_box
    return crop_img(img, x, y, width, height)

# we only retrieve the external contours, as these are the only ones we're interested in;
# we've also used `cv2.CHAIN_APPROX_SIMPLE` to compress the contouring, it reduces the number of data points to process
def get_contoured_img(inverted_binary_img):
    contours, _ = cv2.findContours(inverted_binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_contours_bounding_boxes(contours, min_width=10, min_height=10):
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return [box for box in bounding_boxes if box[2] > min_width and box[3] > min_height]

def invert_colors(binary_img):
    return 255 - binary_img

def load_raw_img(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def save_img(img, path):
    cv2.imwrite(path, img)