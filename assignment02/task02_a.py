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
			'opera' : './images/opera.tiff',
			'aerial' : './images/5.1.10-aerial.tiff'}


def spatialConvolution(inputImage, kernel):
	outputImage = np.array(inputImage, dtype=np.float32)
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
	[-1, -1, -1]], dtype=np.float32)

kernel_highpass_5x5 = np.array([
	[-1, -1, -1, -1, -1],
	[-1,  1,  2,  1, -1],
	[-1,  2,  4,  2, -1],
	[-1,  1,  2,  1, -1],
	[-1, -1, -1, -1, -1]])


def zeroFrequencyComponent(inputImage):
	f = np.fft.fft2(inputImage)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	return magnitude_spectrum


def frequencyDomainFiltering(inputImage):
	f = np.fft.fft2(inputImage)
	fshift = np.fft.fftshift(f)
	rows, cols = inputImage.shape
	crow,ccol = rows/2, cols/2
	fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)
	return img_back


def subplot(filePath, kernel):
	# ndConvolve = ndimage.convolve(misc.imread(filePath), kernel)
	# transformedImage = spatialConvolution(misc.imread(filePath), kernel)
	plt.subplot(121)
	plt.title("original image")
	plt.imshow(misc.imread(filePath), cmap = 'gray')
	plt.axis('off')
	plt.subplot(122)
	magnitude_spectrum = zeroFrequencyComponent(misc.imread(filePath))
	plt.title('magnitude_spectrum')
	plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.axis('off')
	# plt.subplot(223)
	# filteredImage = frequencyDomainFiltering(misc.imread(filePath))
	# plt.title('high pass filtered')
	# plt.imshow(filteredImage)
	plt.show()
	return None

# subplot(imagePath['fishingBoat'], kernel_highpass_3x3)
subplot(imagePath['fishingBoat'], kernel_highpass_5x5)
#plt.imshow(frequencyDomainFiltering(image,kernel_highpass_3x3))

# def gaussian(size=3, std=1.0):
#     s = (size - 1) // 2
#     h = np.linspace(-s, s, size)
#     h = np.exp(-h**2 / (2 * std**2))
#     h = h * h[np.newaxis, :].T
#     sumh = h.sum()
#     if 0.0 != sumh:
#         h /= sumh
#     return h
# print(kernel_highpass_3x3.shape)
# print(kernel_highpass_3x3.size)
# plt.figure()
# plt.subplot(141)
# plt.gray()
# plt.imshow(gaussian())
# plt.axis('off')
# plt.subplot(142)
# plt.gray()
# plt.imshow(gaussian(5, 2.0))
# plt.axis('off')
# plt.subplot(143)
# plt.gray()
# plt.imshow(gaussian(16, 2.0))
# plt.axis('off')
# plt.subplot(144)
# plt.gray()
# plt.imshow(gaussian(128, 20.0))
# plt.axis('off')
# plt.tight_layout()
# plt.show()
# subplotImage(imagePath['fishingBoat'])
