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
	#Makes a copy of the image to write over
	outputImage = np.array(inputImage) 
	#Find height and width of image to help us navigate
	(imageHeight, imageWidth) = inputImage.shape

	#Find height, width and center of kernel to help us navigate
	kernelWidth = len(kernel[0]) 
	centerKernelWidth = int(np.floor(kernelWidth / 2)) 
	kernelHeight = len(kernel) 
	centerKernelHeight = int(np.floor(kernelHeight / 2)) 
	
	
	#traverse through every pixel in the image
	for y in range(imageHeight):
		for x in range(imageWidth):
			#Include a summation of the intensety of the pixel
			#set to 0 whenever we move to the next pixel
			sum = 0
			
			#Traverse through the kernel
			for ky in range(kernelHeight):
				for kx in range(kernelWidth):
					#Find the position of the current kernel square
					#the center of the kernel get position 0.0
					#everything above and to the left of the kernel
					#get negative position
					#The rest are positive
					nx = kx - centerKernelHeight
					ny = ky - centerKernelWidth
				
					#kernel positions are both negative and positive
					#traverse the picture by adding the kernel
					#position and the position of the current pixel
					currentPixelX = x + nx
					currentPixelY = y + ny
				
					#Since we don't have padding are all values 
					#that layes outside the picture excluded
					if 0 <= currentPixelX < imageWidth and 0 <= currentPixelY < imageHeight:
						#Get the intensity of the pixel and multiply with the
						#value in the corresponding square in the kernel
						level = inputImage[currentPixelY] [currentPixelX] * kernel[ky][kx]
						
						#We summarize the intensity of all the kernel pixels
						sum = sum + level
						
			#set the corresponding pixel equal to the new intensity
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
image = spatialConvolution(image, gauss5x5Kernel)
verticalImage = spatialConvolution(image, vertical3x3Kernel)
horizontalImage = spatialConvolution(image, horizontal3x3kernel)
magnitudeImage = np.sqrt(np.power(verticalImage, 2) + np.power(horizontalImage, 2))
ax[1].imshow(magnitudeImage, cmap='gray')
ax[1].set_axis_off()
plt.show()