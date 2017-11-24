import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage

#	create a dictionary for easier filepath handling
imagePath = {'yeast' : './images/Fig1043(a)(yeast_USC).tiff',
			'weld' : './images/Fig1051(a)(defective_weld).tiff',
			'noisy' : './images/noisy.tiff',
			'task5one' : './images/task5-01.tiff',
			'task5two' : './images/task5-02.tiff',
			'task5three' : './images/task5-03.tiff',
			'noisless' : './images/noisless.tiff'}


def detectChessColor(image):
	sum1 = 0
	sum2 = 0
	chessColor1 = [0,0,0]
	chessColor2 = [0,0,0]
	for y in range(5):
		for x in range(7):
			if (image[5+100*y,50+100*x,0]<100):
				chessColor1 += image[5+100*y,50+100*x,:]
				sum1+=1
			else:
				chessColor2 += image[5+100*y,50+100*x,:]
				sum2+=1
	chessColor1 = chessColor1//sum1
	chessColor2 = chessColor2//sum2
	return (chessColor1, chessColor2)


def removeChessBoard(image, color1, color2):
	yImage = np.size(image, 0)
	xImage = np.size(image, 1)
	canvas = np.full((yImage,xImage),1)
	canvas = canvas.astype(bool)
	for y in range(yImage):
		for x in range(xImage):
			canvas[y,x] = not(imageTreshold(image, y, x, color1) or imageTreshold(image, y, x, color2) )
	return canvas


def imageTreshold(image, y, x, color):
	diff = np.absolute(image[y,x] - color)
	treshold = 66
	magnetude = np.sqrt(np.power(diff[0],2)+np.power(diff[1],2)+np.power(diff[2],2))
	#print(magnetude)
	return magnetude < treshold


def removingNoise(image):
	kernel = np.array([[1, 1, 1],
						[1, 1, 1],
						[1, 1, 1]
						])
	canvas = ndimage.binary_opening(image, kernel, iterations=3)
	canvas = ndimage.binary_closing(canvas, kernel, iterations=1)
	return canvas


image = misc.imread(imagePath['task5three'])
originalImage = image

(color1, color2) = detectChessColor(image)

binaryImage = removeChessBoard(image, color1, color2)
binaryImage = removingNoise(binaryImage)

plt.subplot(121)
plt.axis('off')
plt.title('Original image')
plt.imshow(originalImage)
plt.subplot(122)
plt.axis('off')
plt.title('Binary image')
plt.imshow(binaryImage, cmap = 'gray',  interpolation='nearest')

plt.show()
