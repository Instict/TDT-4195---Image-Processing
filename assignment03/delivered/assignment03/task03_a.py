import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image


#	create a dictionary for easier filepath handling
imagePath = {'yeast' : './images/Fig1043(a)(yeast_USC).tiff',
			'weld' : './images/Fig1051(a)(defective_weld).tiff',
			'noisy' : './images/noisy.tiff',
			'task5one' : './images/task5-01.tiff',
			'task5two' : './images/task5-02.tiff',
			'task5three' : './images/task5-03.tiff',
			'noisless' : './images/noisless.png}'}


def makeBinaryImage(image, threshold):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	binaryImage = np.zeros((yImage,xImage), dtype = 'bool')
	for y in range(yImage):
		for x in range(xImage):
			if(image[y,x] > threshold):
				binaryImage[y,x] = 1
	return binaryImage


def createKernel(size):
	kernel = np.full((size,size),0)
	kernel = kernel.astype(bool)
	centerKernel = size//2
	for y in range(size):
		for x in range(size):
			r = np.sqrt(np.power(x-centerKernel,2) + np.power(y-centerKernel,2))
			if (r<centerKernel):
				kernel[y,x] = 1
	return kernel


def erosion(image, kernel):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	print('start Erosion')
	erodedImage = np.full((yImage,xImage),0)
	erodedImage = erodedImage.astype(bool)
	for y in range(yImage):
		for x in range(xImage):
			erodedImage[y,x] = travaserErosion(image, x, y, kernel)
	return erodedImage


def travaserErosion(image, x, y, kernel):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	sizeKernel = np.size(kernel, 0)
	centerKernel = sizeKernel//2
	for j in range(sizeKernel):
		for i in range(sizeKernel):
			if(kernel[j,i]):
				currentPixelY = y - j + centerKernel
				currentPixelX = x - i + centerKernel
				if (0 <= currentPixelY < yImage and 0 <= currentPixelX < xImage ):
					if(image[currentPixelY, currentPixelX] == 0):
						return 0
	return 1


def dilation(image, kernel):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	print('start dilation')
	dilatedImage = np.full((yImage,xImage),0)
	dilatedImage = dilatedImage.astype(bool)
	for y in range(yImage):
		for x in range(xImage):
			dilatedImage[y,x]=traverseDilation(image, x, y, kernel)
	return dilatedImage


def traverseDilation(image, x, y, kernel):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	sizeKernel = np.size(kernel, 0)
	centerKernel = sizeKernel//2
	for j in range(sizeKernel):
		for i in range(sizeKernel):
			if(kernel[j,i]):
				currentPixelY = y - j + centerKernel
				currentPixelX = x - i + centerKernel
				if (0 <= currentPixelY < yImage and 0 <= currentPixelX < xImage ):
					if(image[currentPixelY, currentPixelX] == 1):
						return 1
	return 0


image = misc.imread(imagePath['noisy'])
image = np.array(image, dtype=np.float32)

threshold = 127

binaryImage = makeBinaryImage(image, threshold)

##Burde vere oddetall for best effekt
kernelSize = 15
kernel = createKernel(kernelSize)

plt.plot()
plt.title('Kernel')
plt.axis('off')
plt.imshow(kernel, cmap = 'gray',  interpolation='nearest')
plt.show()

# open operation
openedImage = erosion(binaryImage, kernel)
openedImage = dilation(openedImage, kernel)


plt.subplot(131)
plt.axis('off')
plt.title('Original image')
plt.imshow(binaryImage, cmap = 'gray',  interpolation='nearest')
plt.subplot(132)
plt.axis('off')
plt.title('Opned image')
plt.imshow(openedImage, cmap = 'gray',  interpolation='nearest')

# closing operation
kernel = createKernel(17)
closedImage = dilation(openedImage, kernel)

kernel = createKernel(15)
closedImage = erosion(closedImage, kernel)

plt.subplot(133)
plt.axis('off')
plt.title('Closed image')
plt.imshow(closedImage, cmap = 'gray',  interpolation='nearest')
plt.show()

plt.axis('off')
plt.imshow(closedImage, cmap = 'gray',  interpolation='nearest')
misc.imsave('outfile.tiff', closedImage)
#plt.show()
