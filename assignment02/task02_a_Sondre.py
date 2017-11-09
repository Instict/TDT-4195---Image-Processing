import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from PIL import Image

# Sondre sitt forsok
#	create a dictionary for easier filepath handling
imagePath = {'jellyBeans' : './images/4.1.07-jelly-beans.tiff',
			'fishingBoat' : './images/fishingboat.tiff',
			'lochness' : './images/lochness.tiff',
			'noiseA' : './images/noise-a.tiff',
			'noiseB' : './images/noise-b.tiff',
			'noiseC' : './images/noise-c.tiff',
			'opera' : './images/opera.tiff'}

# A very simple and very narrow highpass filter
kernel_highpass_3x3 = (1/9) * np.array([
	[-1, -1, -1],
	[-1,  8, -1],
	[-1, -1, -1]])
	
kernel_lowpass_3x3 = (1/9) * np.array([
	[1, 1, 1],
	[1, 1, 1],
	[1, 1, 1]])

kernel_highpass_5x5 = np.array([
	[-1, -1, -1, -1, -1],
    [-1,  1,  2,  1, -1],
    [-1,  2,  4,  2, -1],
    [-1,  1,  2,  1, -1],
    [-1, -1, -1, -1, -1]])
	
kernel_lowpass_5x5 = (1/256) * np.array([
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
])

kernel_lowpass_5x5_ones = np.array([
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1]
])

kernel_lowpass_7x7_ones = np.array([
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1]
])

kernel_highpass_7x7 = np.array([
	[-1, -1, -1, -1, -1, -1, -1],
	[-1, -1, -1, -1, -1, -1, -1],
    [-1, -1,  1,  2,  1, -1, -1],
    [-1, -1,  2,  4,  2, -1, -1],
    [-1, -1,  1,  2,  1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
	[-1, -1, -1, -1, -1, -1, -1]
])

kernel_lowpass_9x9_ones = np.array([
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1]
])

kernel_highpass_9x9_ones = np.array([
  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
  [-1, -1,  1,  1,  2,  1,  1, -1, -1],
  [-1, -1,  1,  2,  4,  2,  1, -1, -1],
  [-1, -1,  2,  4,  8,  4,  2, -1, -1],
  [-1, -1,  1,  2,  4,  2,  1, -1, -1],
  [-1, -1,  1,  1,  2,  1,  1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1, -1, -1, -1]
])

kernel_lowpass_7x7 = np.array([
  [1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 2, 1, 1, 1],
  [1, 1, 2, 4, 2, 1, 1],
  [1, 2, 4, 8, 4, 2, 1],
  [1, 1, 2, 4, 2, 1, 1],
  [1, 1, 1, 2, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1]
])

def makeHighpassKernel(n):
	
	kernel = np.array([
			[1, 2, 1],
			[2, 4, 2],
			[1, 2, 1]
					])
	
	kernel = np.lib.pad(kernel, ((n,n), (n, n)), 'constant', constant_values=(-1, -1))
	
	return kernel

def subplot(filePath, kernel):
	image = misc.imread(filePath)
	image = np.array(image, dtype=float)
	ndConvolve = ndimage.convolve(image, kernel)
	image2 = misc.imread(filePath)
	image2 = np.array(image2, dtype=float)
	transformedImage = frequencyDomainFiltering(image2, kernel)
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


def gaussian(size=3, std=1.0):
    s = (size - 1) // 2
    h = np.linspace(-s, s, size)
    h = np.exp(-h**2 / (2 * std**2))
    h = h * h[np.newaxis, :].T
    sumh = h.sum()
    if 0.0 != sumh:
        h /= sumh
    return h
		
def spatial2FrequencyDomain(image, kernel):
	

	
	#Find dim of image
	ySize = np.size(image,0)
	xSize = np.size(image,1)
	
	#Padd the image
	image = np.lib.pad(image, ((ySize,0), (xSize,0)), 'constant', constant_values=(0, 0))
	
	#Centering
	for y in range (2*ySize):
		for x in range (2*xSize):
			image[y,x]=image[y,x]*np.power(-1,x+y)
	
	#Converting to frequency domain
	image_FF = np.fft.fft2(image)
	
	
	
	print('imageY: ',np.size(image_FF,0))
	print('imageX: ',np.size(image_FF,1))


	kernelSize = np.size(kernel,0)
	
	
	kernel = np.lib.pad(kernel, ((ySize-(kernelSize//2),ySize-(kernelSize//2)-1), (xSize-(kernelSize//2), xSize-(kernelSize//2)-1)), 'constant', constant_values=(0, 0))
	kernel_FF = np.fft.fft2(kernel)
	
	print('kernelY: ',np.size(kernel_FF,0))
	print('kernelX: ',np.size(kernel_FF,1))
	
	return(image_FF, kernel_FF)

def matrixMultiplication(F, H):	
	
	G = F
	
	ySize = np.size(F,0)
	xSize = np.size(F,1)
	
	for u in range(ySize):
		for v in range (xSize):
			G[u,v]=H[u,v]*F[u,v]
	
	return G	

def frequency2SpatialDomain(G):
	
	ySize = np.size(G,0)
	xSize = np.size(G,1)
	
	
	print(ySize//2)
	
	g = np.zeros((ySize//2, xSize//2))
	print(g)
	
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
	
	return (g, g_p)
	
def zeroFrequencyComponent(inputImage):
	f = np.fft.fft2(inputImage)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	return magnitude_spectrum

def create_kernel(nxn,q=1):
  k = []
  e = np.e
  c = int(nxn/2)
  q2 = np.power(q,2)
  s = 0
  for u in range(nxn):
    k.append([])
    for v in range(nxn):
      exp = (np.power(u-c,2) + np.power(v-c,2))/(2*q2)
      value = 1/((2*np.pi)*q2*np.power(e,exp))
      k[u].append(value)
      s += value
  print(s)
  K = [[(k[x][y]/s) for y in range(len(k))] for x in range(len(k))]
  return K
	
image = misc.imread('images/fishingboat.tiff')
image = np.array(image, dtype=float)
originalImage = image
#in makeHighpassKernel is the inn comand number of lines to be padded
#if the comand is 2 then a 7x7 matrix is created
kernel = kernel_lowpass_7x7
(image_FF, kernel_FF) = spatial2FrequencyDomain(image, kernel)
ressult = matrixMultiplication(image_FF, kernel_FF)
(ressult2, ressult3) = frequency2SpatialDomain(ressult)
specter = zeroFrequencyComponent(ressult2)
specter2 = zeroFrequencyComponent(originalImage)

plt.subplot(121)
plt.title('ndimage.convolve')
plt.gray()
plt.imshow(specter2)
plt.subplot(122)
plt.gray()
plt.title('Low Pass')
plt.imshow(specter)
plt.show()


_, ax = plt.subplots(1, 3, figsize=(16, 8))

ax[1].imshow(ressult2, cmap='gray')
ax[1].set_axis_off()
ax[0].imshow(originalImage, cmap='gray')
ax[0].set_axis_off()



image = misc.imread('images/fishingboat.tiff')
image = np.array(image, dtype=float)
kernel = kernel_highpass_7x7
(image_FF, kernel_FF) = spatial2FrequencyDomain(image, kernel)
ressult = matrixMultiplication(image_FF, kernel_FF)
(ressult2, ressult3) = frequency2SpatialDomain(ressult)

image2 = originalImage - ressult2

ax[2].imshow(ressult2, cmap='gray')
ax[2].set_axis_off()
plt.show()
	
	#####FIKS SEINARE#######
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
