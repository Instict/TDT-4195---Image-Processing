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
			sum = 0
			sum += rgb[i, j, 0]
			sum += rgb[i, j, 1]
			sum += rgb[i, j, 2]
			averageColor = sum/3
			rgb[i, j] = [averageColor, averageColor, averageColor]
	return rgb
def weightedAverage(rgb):
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
			sum = 0
			luminance = [0.2126, 0.7152, 0.0722]
			sum += rgb[i, j, 0]*luminance[0]
			sum += rgb[i, j, 1]*luminance[1]
			sum += rgb[i, j, 2]*luminance[2]
			rgb[i, j] = [sum, sum, sum]
	return rgb
image = misc.imread('./images/4.1.07-jelly-beans.tiff')
averageImage = average(image)
image = misc.imread('./images/4.1.07-jelly-beans.tiff')
weightedAverageImage = weightedAverage(image)
image = misc.imread('./images/4.1.07-jelly-beans.tiff')
originalImage = image

_, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[2].imshow(originalImage)
ax[2].set_axis_off()
ax[0].imshow(averageImage)
ax[0].set_axis_off()
ax[1].imshow(weightedAverageImage)
ax[1].set_axis_off()
plt.show()