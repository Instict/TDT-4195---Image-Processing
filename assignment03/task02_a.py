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
			'task5three' : './images/task5-03.tiff'}


def spatial2FrequencyDomain(image):
	# fast fourier transform
	F_orig = np.fft.fft2(image)
	# shift image
	F = np.fft.fftshift(F_orig)
	return F


def frequency2SpatialDomain(G):
	ySize = np.size(G,0)
	xSize = np.size(G,1)
	g = np.fft.ifft2(G)
	g = np.real(g)
	#Centering
	for y in range (ySize):
  		for x in range (xSize):
  			g[y,x]=g[y,x]*np.power(-1,x+y)
	return g


def filtering(image):
	for x in range(552,576):
		for y in range(384,386):
			image[y,x] = 0
	for x in range(450,474):
		for y in range(384,386):
			image[y,x] = 0
	return image
	

def flood_fill(image, possitionX, possitionY, seedPointX, seedPointY, threshold, canvas):
	
	if (possitionX < 1 or possitionY < 1):
		
		return canvas
	
	if (image[possitionY,possitionX] == canvas[possitionY,possitionX]):
		return canvas
	
	
	if (np.abs(image[possitionY, possitionX]-image[seedPointY, seedPointX])<threshold):
		canvas[possitionY, possitionX] = image[possitionY, possitionX]
		
		canvas = flood_fill(image, possitionX-1, possitionY-1, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX, possitionY-1, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX+1, possitionY-1, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX-1, possitionY, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX, possitionY, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX+1, possitionY, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX-1, possitionY+1, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX, possitionY+1, seedPointX, seedPointY, threshold, canvas)
		canvas = flood_fill(image, possitionX+1, possitionY+1, seedPointX, seedPointY, threshold, canvas)
	
	return canvas

image = misc.imread(imagePath['weld'])
image = np.array(image, dtype=np.float32)

imageSeedPoint1 = image
seedPointX = 140
seedPointY = 255
threshold = 50

canvas = np.zeros((np.size(image,0),np.size(image, 1)))
canvas = flood_fill(image, seedPointX, seedPointY, seedPointX, seedPointY, threshold, canvas)

seedPointX = 294
seedPointY = 252
canvas = flood_fill(image, seedPointX, seedPointY, seedPointX, seedPointY, threshold, canvas)

seedPointX = 359
seedPointY = 240
canvas = flood_fill(image, seedPointX, seedPointY, seedPointX, seedPointY, threshold, canvas)

seedPointX = 414
seedPointY = 234
canvas = flood_fill(image, seedPointX, seedPointY, seedPointX, seedPointY, threshold, canvas)

seedPointX = 444
seedPointY = 243
canvas = flood_fill(image, seedPointX, seedPointY, seedPointX, seedPointY, threshold, canvas)

plt.imshow(canvas)
plt.gray()
plt.show()




# # convert to frequency
# F = spatial2FrequencyDomain(image)
# plt.subplot(121)
# plt.axis('off')
# plt.title('Spectrum of image')
# plt.imshow(np.log(1+np.abs(F)), cmap = 'gray')

# # filter image
# filtered = filtering(F)
# plt.subplot(122)
# plt.axis('off')
# plt.title('Spectrum after filter')
# plt.imshow(np.log(1+np.abs(filtered)), cmap = 'gray')
# plt.show()

# # convert to spatial
# g = frequency2SpatialDomain(filtered)
# plt.subplot(121)
# plt.axis('off')
# plt.title('Unfiltered image')
# plt.imshow(image, cmap = 'gray')
# plt.subplot(122)
# plt.axis('off')
# plt.title('Filtered image')
# plt.imshow(g, cmap = 'gray')
# plt.show()
