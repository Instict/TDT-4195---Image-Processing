import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
import matplotlib.image as mpimg
from scipy import misc
from PIL import Image


#	load an image an display it using matplotlib
image = misc.imread('./lochness.tiff')

#	greyscale conversion
def average(rgb):
	return ((rgb[0] + rgb[1] + rgb[2]) / 3)

def weightedAverage(rgb):
	return 0.2125*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]

greyAverage = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
greyWeightedAverage = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
# get row number
for rownum in range(len(image)):
	for colnum in range(len(image[rownum])):
		greyAverage[rownum][colnum] = average(image[rownum][colnum])
		greyWeightedAverage[rownum][colnum] = weightedAverage(image[rownum][colnum])

		
#plt.figure(1)
#plt.title('Using matplotlib')
#plt.imshow(grey, cmap = plt.cm.Greys_r)
#plt.show()

_, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(greyAverage, cmap = plt.cm.Greys_r)  # Red colour map
ax[0].set_axis_off()
ax[1].imshow(greyWeightedAverage, cmap = plt.cm.Greys_r)  # Red colour map
ax[1].set_axis_off()
plt.show()