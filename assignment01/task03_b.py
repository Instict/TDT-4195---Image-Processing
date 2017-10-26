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

#	gamma transformations
def gammaTransform(rgb,gammaValues):
	for i in range(len(rgb)):
		for j in range(len(rgb[i])):
		#	normalize the image
			p = rgb[i, j]/255
			rgb[i, j] = (p**gammaValues)*255
	return rgb

def subplotImage(filepath):
	_, ax = plt.subplots(1, 3, figsize=(16, 8))
#	store gamma values in array
	gammaValues = [.2, .5, .9]
#	read from array and create plot for each value
	for i in range(len(gammaValues)):
		image = misc.imread(filepath)	
		gammaTransformed = gammaTransform(image,gammaValues[i])
		ax[i].imshow(gammaTransformed, cmap=plt.cm.gray)
		ax[i].set_axis_off()
	plt.show()
	return None

subplotImage(imagePath['aerial'])
subplotImage(imagePath['fishingBoat'])