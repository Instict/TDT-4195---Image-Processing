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
	
def erosion(image, kernel):
	
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)

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
	
def boundary(image, erodedImage)
	
	boundaryImage = image - erodedImage
	
	retun boundaryImage
	
kernel =  np.array([
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
])	
	
image = misc.imread(imagePath['noisless'])
image = np.array(image, dtype=np.float32)

threshold = 127

binaryImage = makeBinaryImage(image, threshold)

erodedImage = erosion(binaryImage, kernel)

boundaryImage = boundary(image, erodedImage)

plt.subplot(131)
plt.axis('off')
plt.title('Original image')
plt.imshow(binaryImage, cmap = 'gray',  interpolation='nearest')
plt.subplot(132)
plt.axis('off')
plt.title('Eroded image')
plt.imshow(erodedImage, cmap = 'gray',  interpolation='nearest')
plt.subplot(133)
plt.axis('off')
plt.title('Closed image')
plt.imshow(boundaryImage, cmap = 'gray',  interpolation='nearest')

plt.show()

