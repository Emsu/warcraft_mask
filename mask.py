import dlib
import sys
import cv2
import numpy as np
from PIL import Image


BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
MASK_PATH = "./input_mask.png"
MASK_COLOR = (0.0, 0.0, 0.0)
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MASK_FACE_WIDTH = 400.0
MASK_IMAGE_WIDTH = 632.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_face_landmarks(input_image):
    image = np.array(input_image)
    rects = detector(image, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.array([[p.x, p.y] for p in
                    predictor(image, rects[0]).parts()])


def get_face_quad(landmarks):
    top_left = landmarks[0]
    bottom_left = landmarks[4]
    bottom_right = landmarks[12]
    top_right = landmarks[16]

    # Find min rect that fits all four points
    left = min(top_left[0], bottom_left[0])
    upper = min(top_left[1], top_right[1])
    right = max(top_right[0], bottom_right[0])
    bottom = min(bottom_left[1], bottom_right[1])
    return left, upper, right, bottom


def get_mask():
    return load_image(MASK_PATH)


def create_background(mask):
    """ Create black background image """
    return Image.new(mask.mode, mask.size, "black")


def load_image(path):
    return Image.open(path)


def get_center_from_quad(left, upper, right, bottom):
    return (left + ((right - left) / 2), upper + ((bottom - upper) / 2))


def quad_from_center_sides(center, width, height):
    x, y = center
    left = x - width / 2
    right = x + width / 2
    upper = y - height / 2
    bottom = y + height / 2
    return left, upper, right, bottom


def crop_to_face(face, cropping_layer, landmarks):
    left, upper, right, bottom = get_face_quad(landmarks)
    scale_factor = MASK_FACE_WIDTH / float(right - left)
    copy = face.copy()
    resized_face = copy.resize((int(round(face.size[0] * scale_factor)), int(round(face.size[1] * scale_factor))))
    landmarks = get_face_landmarks(resized_face)
    center = get_center_from_quad(*get_face_quad(landmarks))
    width, height = cropping_layer.size

    cropped = resized_face.crop(quad_from_center_sides(center, width, height))
    cropped.putalpha(255)  # convert RGB to RGBA for melding
    return cropped


def meld_layers(input_layer, melding_layer, landmarks):
    cropped_face = crop_to_face(input_layer, melding_layer, landmarks)
    # img = correct_colours(np.array(cropped_face), np.array(melding_layer), landmarks)
    copy = cropped_face.copy()
    copy.paste(melding_layer, (0, 0), melding_layer)
    return copy

def get_holes(image, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    im_bw_inv = cv2.bitwise_not(im_bw)

    _, contour, _ = cv2.findContours(im_bw_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(im_bw_inv, [cnt], 0, 255, -1)

    nt = cv2.bitwise_not(im_bw)
    im_bw_inv = cv2.bitwise_or(im_bw_inv, nt)
    return im_bw_inv


def remove_background(image, thresh, scale_factor=.25, kernel_range=range(1, 15), border=None):
    image = np.array(image)
    border = border or kernel_range[-1]

    holes = get_holes(image, thresh)
    small = cv2.resize(holes, None, fx=scale_factor, fy=scale_factor)
    bordered = cv2.copyMakeBorder(small, border, border, border, border, cv2.BORDER_CONSTANT)

    for i in kernel_range:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*i+1, 2*i+1))
        bordered = cv2.morphologyEx(bordered, cv2.MORPH_CLOSE, kernel)

    unbordered = bordered[border: -border, border: -border]
    mask = cv2.resize(unbordered, (image.shape[1], image.shape[0]))
    fg = cv2.bitwise_and(image, image, mask=mask)
    return Image.fromarray(fg)


class Quad(object):

    def __init__(self):
        pass

    def to_rect(self):
        pass


class Rect(object):

    def __init__(self, left, upper, right, bottom):
        self.left = left
        self.upper = upper
        self.right = right
        self.bottom = bottom

    def to_points(self):
        return (self.left, self.upper, self.right, self.bottom)

    def to_sides(self):
        pass


# Actual Script
if len(sys.argv) != 2:
    raise Exception("Please pass in path to face image")
image_path = sys.argv[1]
face_image = load_image(image_path)
mask = get_mask()
background = create_background(mask.copy())

landmarks = get_face_landmarks(face_image.copy())
quad = get_face_quad(landmarks)
new_face = remove_background(face_image.copy(), 170)

masked_face = meld_layers(new_face.copy(), mask.copy(), landmarks)
background.paste(masked_face.copy(), (0, 0), masked_face)
background.save("out.png")
