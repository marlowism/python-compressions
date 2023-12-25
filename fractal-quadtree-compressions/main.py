from quadtree import *
from affine import *

image_path = "input.jpg"
image = Image.open(image_path)




#quadtree = QuadTree(image)

depth = 10
#image = quadtree.create_image(depth, show_lines=False)
#image.save("out.jpg")

fractal_compression()