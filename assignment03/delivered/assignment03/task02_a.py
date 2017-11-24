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
originalImage = image

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

plt.subplot(121)
plt.title('Original image')
plt.axis('off')
plt.imshow(originalImage, cmap = 'gray')
plt.subplot(122)
plt.title('Segmented image')
plt.axis('off')
plt.imshow(canvas, cmap = 'gray')
plt.show()
