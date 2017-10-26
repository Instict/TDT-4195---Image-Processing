#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

originalImage = misc.imread('./images/lochness.tiff')

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
			sumr = 0
			sumg = 0
			sumb = 0
		
			for ky in range(kernelHeight):
				for kx in range(kernelWidth):
					lx = kx - centerKernelHeight
					ly = ky - centerKernelWidth
				
					currentPixelX = x + lx
					currentPixelY = y + ly
				
					if 0 <= currentPixelX < imageWidth and 0 <= currentPixelY < imageHeight:
						r = inputImage[currentPixelY] [currentPixelX] [0] * kernel[ky][kx]
						g = inputImage[currentPixelY] [currentPixelX] [1] * kernel[ky][kx]
						b = inputImage[currentPixelY] [currentPixelX] [2] * kernel[ky][kx]
					
						sumr = sumr + r
						sumg = sumg + g
						sumb = sumb + b
					
			outputImage[y][x][0] = sumr
			outputImage[y][x][1] = sumg
			outputImage[y][x][2] = sumb
			
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

filteredImage = spatialConvolution(originalImage, gaussian5x5Kernel)
print("Før bilde")
_, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[1].imshow(filteredImage)
ax[1].set_axis_off()
ax[0].imshow(originalImage)
ax[0].set_axis_off()
plt.show()
print("Etter bilde")