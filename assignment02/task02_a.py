import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from PIL import Image

#	create a dictionary for easier filepath handling
imagePath = {'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
			'fishingBoat' : './images/fishingboat.tiff',
			'lochness' : './images/lochness.tiff',
			'noiseA' : './images/noise-a.tiff',
			'noiseB' : './images/noise-b.tiff',
			'noiseC' : './images/noise-c.tiff',
			'opera' : './images/opera.tiff'}

def frequencyDomainFiltering(inputImage, kernel):
	#	do FFT math
	#transformedImage = ndimage.convolve(grayscale, kernel)
	#inputImage = np.fft.fft2(inputImage, kernel)
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
						sum += level
			outputImage[y][x] = sum
	return outputImage


# A very simple and very narrow highpass filter
kernel_highpass_3x3 = np.array([
	[-1, -1, -1],
	[-1,  8, -1],
	[-1, -1, -1]])

kernel_highpass_5x5 = np.array([
	[-1, -1, -1, -1, -1],
	[-1,  1,  2,  1, -1],
	[-1,  2,  4,  2, -1],
	[-1,  1,  2,  1, -1],
	[-1, -1, -1, -1, -1]])

def subplot(filePath, kernel):
	ndConvolve = ndimage.convolve(misc.imread(filePath), kernel)
	transformedImage = frequencyDomainFiltering(misc.imread(filePath), kernel)
	plt.subplot(121)
	plt.title('ndimage.convolve')
	plt.gray()
	plt.imshow(ndConvolve)
	plt.subplot(122)
	plt.gray()
	plt.title('High Pass')
	plt.imshow(transformedImage)
	plt.show()
	return None

subplot(imagePath['fishingBoat'], kernel_highpass_3x3)
subplot(imagePath['fishingBoat'], kernel_highpass_5x5)
#plt.imshow(frequencyDomainFiltering(image,kernel_highpass_3x3))

def gaussian(size=3, std=1.0):
    s = (size - 1) // 2
    h = np.linspace(-s, s, size)
    h = np.exp(-h**2 / (2 * std**2))
    h = h * h[np.newaxis, :].T
    sumh = h.sum()
    if 0.0 != sumh:
        h /= sumh
    return h
print(kernel_highpass_3x3.shape)
print(kernel_highpass_3x3.size)
plt.figure()
plt.subplot(141)
plt.gray()
plt.imshow(gaussian())
plt.axis('off')
plt.subplot(142)
plt.gray()
plt.imshow(gaussian(5, 2.0))
plt.axis('off')
plt.subplot(143)
plt.gray()
plt.imshow(gaussian(16, 2.0))
plt.axis('off')
plt.subplot(144)
plt.gray()
plt.imshow(gaussian(128, 20.0))
plt.axis('off')
plt.tight_layout()
plt.show()
# subplotImage(imagePath['fishingBoat'])
