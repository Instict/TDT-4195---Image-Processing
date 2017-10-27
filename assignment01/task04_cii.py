import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

imagePath = { 'aerial' : './images/5.1.10-aerial.tiff',
			'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
			'lake' : './images/4.2.06-lake.tiff',
			'fishingBoat' : './images/fishingboat.tiff',
			'lochness' : './images/lochness.tiff',
			'terraux' : './images/terraux.tiff'}

image = misc.imread(imagePath['fishingBoat'])
image = image.astype('float32') 			

def spatialConvolution(inputImage, kernel):
	
	outputImage = np.array(inputImage) 
	(imageHeight, imageWidth) = inputImage.shape

	kernelWidth = len(kernel[0]) 
	centerKernelWidth = int(np.floor(kernelWidth / 2)) 
	kernelHeight = len(kernel) 
	centerKernelHeight = int(np.floor(kernelHeight / 2)) 
	
	
	
	for y in range(imageHeight):
		for x in range(imageWidth):
			sum = 0
			
			for ky in range(kernelHeight):
				for kx in range(kernelWidth):
					nx = kx - centerKernelHeight
					ny = ky - centerKernelWidth
				
					currentPixelX = x + nx
					currentPixelY = y + ny
				
					if 0 <= currentPixelX < imageWidth and 0 <= currentPixelY < imageHeight:
						level = inputImage[currentPixelY] [currentPixelX] * kernel[ky][kx]
						sum = sum + level
						
			outputImage[y][x] = sum
			
	return outputImage


horizontal3x3kernel = np.array([
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
])

vertical3x3Kernel = np.array([
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
])

gauss5x5Kernel = 1/256 * np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
])

_, ax = plt.subplots(1, 2, figsize=(16, 8))
originalImage = image
ax[0].imshow(originalImage, cmap='gray')
ax[0].set_axis_off()
#image = spatialConvolution(image, gauss5x5Kernel)
verticalImage = spatialConvolution(image, vertical3x3Kernel)
horizontalImage = spatialConvolution(image, horizontal3x3kernel)
magnitudeImage = np.sqrt(np.power(verticalImage, 2) + np.power(horizontalImage, 2))
ax[1].imshow(magnitudeImage, cmap='gray')
ax[1].set_axis_off()
plt.show()