import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
from scipy import misc
from PIL import Image

importImages = { 'aerial' : './images/5.1.10-aerial.tiff',
				'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
				'lake' : './images/4.2.06-lake.tiff',
				'fishingBoat' : './images/fishingboat.tiff',
				'lochness' : './images/fishingboat.tiff',
				'terraux' : './images/terraux.tiff'}

#	intensity transformations
def intensity(rgb):
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
			p = rgb[i, j]
			p_k = 15
			rgb[i, j] = p_k - p
	return rgb

def plotImage(filepath):
	image = misc.imread(filepath)
	intensityImage = intensity(image)
	plt.imshow(intensityImage, cmap=plt.cm.gray)
	plt.axis('off')
	plt.show()
	return None

plotImage(importImages['lochness'])