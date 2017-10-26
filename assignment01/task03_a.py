import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
from scipy import misc
from PIL import Image

imagePath = { 'aerial' : './images/5.1.10-aerial.tiff',
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
			p_k = 255
			rgb[i, j] = p_k - p
	return rgb

def plotImage(filepath):
	image = misc.imread(filepath)
	intensityImage = intensity(image)
	plt.imshow(intensityImage, cmap=plt.cm.gray)
	plt.axis('off')
	plt.show()
	return None
def subplotImage(filepath):
	_, ax = plt.subplots(1, 2, figsize=(16, 8))
	originalImage = misc.imread(filepath)
	ax[0].imshow(originalImage, cmap=plt.cm.gray)
	ax[0].set_axis_off()
	intensityImage = intensity(misc.imread(filepath))
	ax[1].imshow(intensityImage, cmap=plt.cm.gray)
	ax[1].set_axis_off()
	plt.show()
	return None
	
plotImage(imagePath['aerial'])
subplotImage(imagePath['aerial'])