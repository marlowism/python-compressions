import numpy as np
from PIL import Image, ImageDraw

MAX_DEPTH = 10             # max = 10
DETAIL_THRESHOLD = 4


def average_brightness(image):
    image_arr = np.asarray(image)

    avg_brightness = np.mean(image_arr)

    return int(avg_brightness)


def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error


def get_brightness_detail(hist):
    detail_intensity = weighted_average(hist)
    return detail_intensity


class Quadrant():
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_brightness_detail(hist)
        self.colour = average_brightness(image)

    def split_quadrant(self, image):
        left, top, width, height = self.bbox

        middle_x = left + (width - left) / 2
        middle_y = top + (height - top) / 2

        upper_left = Quadrant(image, (left, top, middle_x, middle_y), self.depth + 1)
        upper_right = Quadrant(image, (middle_x, top, width, middle_y), self.depth + 1)
        bottom_left = Quadrant(image, (left, middle_y, middle_x, height), self.depth + 1)
        bottom_right = Quadrant(image, (middle_x, middle_y, width, height), self.depth + 1)

        self.children = [upper_left, upper_right, bottom_left, bottom_right]


class QuadTree():
    def __init__(self, image):
        self.width, self.height = image.size
        self.max_depth = 0
        self.start(image)

    def start(self, image):
        self.root = Quadrant(image, image.getbbox(), 0)
        self.build(self.root, image)

    def build(self, root, image):
        if root.depth >= MAX_DEPTH or root.detail <= DETAIL_THRESHOLD:
            if root.depth > self.max_depth:
                self.max_depth = root.depth

            root.leaf = True
            return

        root.split_quadrant(image)

        for child in root.children:
            self.build(child, image)

    def create_image(self, custom_depth, show_lines=False):

        image = Image.new('L', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), 0)

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.colour, outline=0)
            else:
                draw.rectangle(quadrant.bbox, quadrant.colour)

        return image

    def get_leaf_quadrants(self, depth):

        quadrants = []

        self.recursive_search(self, self.root, depth, quadrants.append)

        return quadrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)