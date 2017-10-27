#!/usr/bin/python3

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




# The main function for applying a filter to an 2d-rgb-array
def spatialConvolution(inputImage, kernel):
	#inputImage = inputImage.astype('float32') # Convert to float to avoid overflow issues
	inputImage = np.array(inputImage)
	outputImage = np.array(inputImage) # Create a copy which can be modified
	(imageHeight, imageWidth, channele) = inputImage.shape

	kernelWidth = len(kernel[0]) # Width of kernel
	centerKernelWidth = int(np.floor(kernelWidth / 2)) # Center width of kernel
	kernelHeight = len(kernel) # Height of kernel
	centerKernelHeight = int(np.floor(kernelHeight / 2)) # Center height of kernel
	
	
	
	for y in range(imageHeight):
		for x in range(imageWidth):
			sumR = 0
			sumG = 0
			sumB = 0
		
			for ky in range(kernelHeight):
				for kx in range(kernelWidth):
					nx = kx - centerKernelHeight
					ny = ky - centerKernelWidth
				
					currentPixelX = x + nx
					currentPixelY = y + ny
				
					if 0 <= currentPixelX < imageWidth and 0 <= currentPixelY < imageHeight:
						levelR = inputImage[currentPixelY] [currentPixelX] [0] * kernel[ky][kx]
						levelG = inputImage[currentPixelY] [currentPixelX] [1] * kernel[ky][kx]
						levelB = inputImage[currentPixelY] [currentPixelX] [2] * kernel[ky][kx]
					
						sumR = sumR + levelR
						sumG = sumG + levelG
						sumB = sumB + levelB
					
			outputImage[y][x][0] = sumR
			outputImage[y][x][1] = sumG
			outputImage[y][x][2] = sumB
			
	return outputImage

avaraging3x3kernel =  (1 / 9) * np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
])

gaussian5x5Kernel =  (1 / 256) * np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
])

def subplotImage(filepath):
	_, ax = plt.subplots(1, 3, figsize=(16, 8))
	originalImage = misc.imread(filepath)
	ax[0].imshow(originalImage, cmap='gray')
	ax[0].set_axis_off()
	avaragingImage = spatialConvolution(misc.imread(filepath), avaraging3x3kernel)
	ax[1].imshow(avaragingImage, cmap='gray')
	ax[1].set_axis_off()
	gaussianImage = spatialConvolution(misc.imread(filepath), gaussian5x5Kernel)
	ax[2].imshow(gaussianImage, cmap='gray')
	ax[2].set_axis_off()
	plt.show()
	return None
	
subplotImage(imagePath['lochness'])