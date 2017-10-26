import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
from scipy import misc
from PIL import Image

#	create a dictionary for easier filepath handling
imagePath = { 'aerial' : './images/5.1.10-aerial.tiff',
			'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
			'lake' : './images/4.2.06-lake.tiff',
			'fishingBoat' : './images/fishingboat.tiff',
			'lochness' : './images/fishingboat.tiff',
			'terraux' : './images/terraux.tiff'}

#	intensity transformations
def intensity(grayscale):
#	loop through the colors
	for i in range(len(grayscale)):
		for j in range(len(grayscale[i])):
		#	save original image
			p = grayscale[i, j]
		#	p_k is the highest value a pixel can have
			p_k = 255
		#	use formula T(p)=p_k-p and update the colors
			grayscale[i, j] = p_k - p
	return grayscale

#	create function to plot and modify the image
def plotImage(filepath):
#	read image
	image = misc.imread(filepath)
	intensityImage = intensity(image)
#	show image in plot, using gray color map 
	plt.imshow(intensityImage, cmap=plt.cm.gray)
	plt.axis('off')
	plt.show()
	return None
	
#	create function for comparing original and modified image
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
	
#	create two plots for the image
plotImage(imagePath['aerial'])
subplotImage(imagePath['aerial'])