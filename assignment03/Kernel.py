import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

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


	
kernel =  np.array([
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,],
  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,],
  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,],
  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,],
  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,],
  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
])
	
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

kernel = createKernel(25)
	
plt.imshow(kernel, cmap = 'gray',  interpolation='nearest')
plt.show()