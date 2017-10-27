imagePath = { 'aerial' : './images/5.1.10-aerial.tiff',
		'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
		'lake' : './images/4.2.06-lake.tiff',
		'fishingBoat' : './images/fishingboat.tiff',
		'lochness' : './images/fishingboat.tiff',
		'terraux' : './images/terraux.tiff'}


def subplotImage(filepath):
	_, ax = plt.subplots(1, 3, figsize=(16, 8))
	originalImage = misc.imread(filepath)
	ax[0].imshow(originalImage, cmap='gray')
	ax[0].set_axis_off()
	avaragingImage = spatialConvolution(misc.imread(filepath), avaraging3x3kernel)
	ax[1].imshow(avaragingImage, cmap='gray')
	ax[1].set_axis_off()
	gaussianImage = spatialConvolution(misc.imread(filepath), gaussian5x5Kernel )
	ax[2].imshow(gaussianImage, cmap='gray')
	ax[2].set_axis_off()
	plt.show()
	return None
	
subplotImage(imagePath['fishingBoat'])
subplotImage(imagePath['aerial'])
