import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

#	create a dictionary for easier filepath handling
imagePath = {'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
			'fishingBoat' : './images/fishingboat.tiff',
			'lochness' : './images/lochness.tiff',
			'noiseA' : './images/noise-a.tiff',
			'noiseB' : './images/noise-b.tiff',
			'noiseC' : './images/noise-c.tiff',
			'opera' : './images/opera.tiff'}


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


image = misc.imread(imagePath['noiseA'])
image = np.array(image, dtype=np.float32)

# convert to frequency
F = spatial2FrequencyDomain(image)
plt.subplot(121)
plt.axis('off')
plt.title('Spectrum of image')
plt.imshow(np.log(1+np.abs(F)), cmap = 'gray')

# filter image
filtered = filtering(F)
plt.subplot(122)
plt.axis('off')
plt.title('Spectrum after filter')
plt.imshow(np.log(1+np.abs(filtered)), cmap = 'gray')
plt.show()

# convert to spatial
g = frequency2SpatialDomain(filtered)
plt.subplot(121)
plt.axis('off')
plt.title('Unfiltered image')
plt.imshow(image, cmap = 'gray')
plt.subplot(122)
plt.axis('off')
plt.title('Filtered image')
plt.imshow(g, cmap = 'gray')
plt.show()
