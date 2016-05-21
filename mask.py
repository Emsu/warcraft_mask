import dlib
import numpy
from PIL import Image

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
MASK_PATH = "./mask.png"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_face_landmarks(image):
    rects = detector(image, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in
                        predictor(image, rects[0]).parts()])


def get_face_quad(image):
    landmarks = get_face_landmarks(image)
    top_left = landmarks[0]
    bottom_left = landmarks[4]
    bottom_right = landmarks[12]
    top_right = landmarks[16]
    return top_left, bottom_left, bottom_right, top_right


def get_mask():
    return load_image(MASK_PATH)


def create_background(mask):
    """ Create black background image """
    return Image.new(mask.mode, mask.size, "black")


def load_image(path):
    return Image.open(path)


def meld_layers(input_layer, melding_layer):
    pass


def remove_background(image):
    """ Removes background from image """
    pass


class Quad(object):

    def __init__(self):
        pass

    def to_rect(self):
        pass


class Rect(object):

    def __init__(self):
        pass

    def to_points(self):
        pass

    def to_sides(self):
        pass


def resize_face_to_mask(face_quad, face_image, mask_image):
    """ Crop out face and resize to fit mask """
    # rotate using quad
    # find center using quad
    # crop to size of mask
    rect = Rect()
    face_image.crop(rect.to_points)
    return remove_background  # (image)


# Actual Script
image_path = 'path from user'
face_image = load_image(image_path)
mask = get_mask()
background = create_background(mask)

quad = get_face_quad(face_image)
new_face = resize_face_to_mask(face_quad=quad,
                               face_image=face_image,
                               mask_image=mask)
masked_face = meld_layers(new_face, mask)
_, _, _, alpha = masked_face.split()
background.paste(masked_face, mask=alpha)
background.save("out.png")
