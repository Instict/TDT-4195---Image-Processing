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

image = misc.imread(imagePath['noiseA'])
image = np.array(image, dtype=np.float32)

spec_orig = np.fft.fft2(image)
spec_img = np.fft.fftshift(spec_orig)

for j in range(115,125):
    for n in range(96,106):
        spec_img[n,j] = 0

def frequency2SpatialDomain(G):
	ySize = np.size(G,0)
	xSize = np.size(G,1)
	g_p = np.fft.ifft2(G)
	g_p = np.real(g_p)
	#Centering
	for y in range (ySize):
		for x in range (xSize):
			g_p[y,x]=g_p[y,x]*np.power(-1,x+y)
	return g_p

plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap = 'gray')
plt.subplot(132)
plt.axis('off')
plt.title('spectrum')
plt.imshow(np.log(1+np.abs(spec_img)), cmap = 'gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
spec_img = frequency2SpatialDomain(spec_img)
plt.imshow(spec_img, cmap = 'gray')
plt.show()
