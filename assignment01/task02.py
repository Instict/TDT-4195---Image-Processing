import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image


#	load an image an display it using matplotlib
image = misc.imread('./lochness.tiff')
plt.figure(1)
plt.title('Using matplotlib')
plt.imshow(image, interpolation='nearest')
plt.show()


#	loaded using NumPy array using PIL
img = Image.open('./lochness.tiff')
img = np.array(img)
print(np.array_equal(image, img))
plt.figure(2)
plt.title('Using NumPy array using PIL')
plt.imshow(img, interpolation='nearest')
plt.show()