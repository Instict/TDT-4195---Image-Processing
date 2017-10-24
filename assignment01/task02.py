import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
import matplotlib.image as mpimg
from scipy import misc
from PIL import Image

#	greyscale conversion
def average(rgb):
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
			grey = 0
			grey += rgb[i, j, 0]
			grey += rgb[i, j, 1]
			grey += rgb[i, j, 2]
			avgColor = grey/3
			rgb[i, j] = [avgColor, avgColor, avgColor]
	return rgb
def weightedAverage(rgb):
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
			grey = 0
			luminance = [0.2126, 0.7152, 0.0722]
			grey += rgb[i, j, 0]*luminance[0]
			grey += rgb[i, j, 1]*luminance[1]
			grey += rgb[i, j, 2]*luminance[2]
			rgb[i, j] = [grey, grey, grey]
	return rgb
#	converting images
image = misc.imread('./images/4.2.06-lake.tiff')
averageImage = average(image)
image = misc.imread('./images/4.2.06-lake.tiff')
weightedAverageImage = weightedAverage(image)
image = misc.imread('./images/4.2.06-lake.tiff')
originalImage = image
#	plots
plt.imshow(originalImage)
plt.axis('off')
plt.show()
plt.imshow(averageImage)
plt.axis('off')
plt.show()
plt.imshow(weightedAverageImage)
plt.axis('off')
plt.show()
#	subplots
_, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[2].imshow(originalImage)
ax[2].set_axis_off()
ax[0].imshow(averageImage)
ax[0].set_axis_off()
ax[1].imshow(weightedAverageImage)
ax[1].set_axis_off()
plt.show()