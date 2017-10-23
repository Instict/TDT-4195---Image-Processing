import numpy as np
#	matplotlib inline
import matplotlib.pyplot as plt
from scipy import misc

from PIL import Image

image = misc.imread('./lochness.tiff')

img = Image.open('./lochness.tiff'')
img = np.array(img)

print(np.array_equal(image, img))