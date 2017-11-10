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
			'opera' : './images/opera.tiff',
			'aerial' : './images/5.1.10-aerial.tiff'}

# A very simple and very narrow highpass filter
kernel_highpass_3x3 = (1/9) * np.array([
	[-1, -1, -1],
	[-1,  8, -1],
	[-1, -1, -1]], dtype=np.float32)


kernel_highpass_5x5 = np.array([
	[-1, -1, -1, -1, -1],
	[-1,  1,  2,  1, -1],
	[-1,  2,  4,  2, -1],
	[-1,  1,  2,  1, -1],
	[-1, -1, -1, -1, -1]])


kernel_Gausian_5x5 = (1/256) * np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]])


def spatial2FrequencyDomain(image, kernel):
	#Find dim of image
	ySize = np.size(image,0)
	xSize = np.size(image,1)
	#Padd the image
	image = np.lib.pad(image, ((ySize,0), (xSize,0)), 'constant', constant_values=(0, 0))
	kernelSize = np.size(kernel,0)
	kernel = np.lib.pad(kernel, ((ySize-(kernelSize//2),ySize-(kernelSize//2)-1), (xSize-(kernelSize//2), xSize-(kernelSize//2)-1)), 'constant', constant_values=(0, 0))
	#Centering
	for y in range (2*ySize):
		for x in range (2*xSize):
			image[y,x]=image[y,x]*np.power(-1,x+y)
			kernel[y,x]=kernel[y,x]*np.power(-1,x+y)
	#Converting to frequency domain
	F = np.fft.fft2(image)
	H = np.fft.fft2(kernel)
	return(F, H)


def frequency2SpatialDomain(G):
	ySize = np.size(G,0)
	xSize = np.size(G,1)
	g = np.zeros((ySize//2, xSize//2))
	g_p = np.fft.ifft2(G)
	g_p = np.real(g_p)
	#Centering
	for y in range (ySize):
		for x in range (xSize):
			g_p[y,x]=g_p[y,x]*np.power(-1,x+y)
	#Restore padded image
	for y in range (ySize//2):
		for x in range (xSize//2):
			g[y,x] = g_p[y,x]
	return g

def unsharpning_kernel(kernel):
	#Finner størrelsen på array
	size = np.size(kernel,0)
	#Lager 1 tallet som skal vare i mitten av impulsen
	impuls = np.array([[1]])
	#Padder den med nuller slik at den blir like stor som kernelen
	impuls = np.lib.pad(impuls, ((size//2,size//2), (size//2, size//2)), 'constant', constant_values=(0, 0))
	#MATH
	kernel = impuls + (impuls - kernel)
	return kernel

#Her kjore eg programmet

image = misc.imread('images/fishingboat.tiff')
image = np.array(image, dtype=float)
original_image = image
kernel = kernel_Gausian_5x5
kernel = unsharpning_kernel(kernel)
(F, H) = spatial2FrequencyDomain(image, kernel)
G = H*F
(g) = frequency2SpatialDomain(G)

plt.subplot(131)
plt.gray()
plt.title('Original image')
plt.imshow(original_image)
plt.subplot(132)
plt.gray()
plt.title('Sharp image')
plt.imshow(g)

# Samanlikne med smooth
image = misc.imread('images/fishingboat.tiff')
image = np.array(image, dtype=float)
kernel = kernel_Gausian_5x5
(F, H) = spatial2FrequencyDomain(image, kernel)
G = H*F
(g2) = frequency2SpatialDomain(G)

plt.subplot(133)
plt.gray()
plt.title('Sharp image')
plt.imshow(g2)
plt.show()
