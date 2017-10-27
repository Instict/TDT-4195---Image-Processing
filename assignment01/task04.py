#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

image = misc.imread('./images/fishingboat.tiff')
#image = misc.imread('./images/rainbow.tiff')
#image = misc.imread('./images/lochness.tiff')
#image = misc.imread('./images/doge.tiff')

# The main function for applying a filter to an 2d-rgb-array
def spatialConvolution(inputImage, kernel):
  inputImage = inputImage.astype('float32') # Convert to float to avoid overflow issues
  outputImage = np.array(inputImage) # Create a copy which can be modified
  (imgHeight, imgWidth) = inputImage.shape

  w = len(kernel[0]) # Width of kernel
  cw = int((w - 1) / 2) # Center width of kernel
  h = len(kernel) # Height of kernel
  ch = int((h - 1) / 2) # Center height of kernel

  for y in range(imgHeight):
    for x in range(imgWidth):
      sr, sg, sb = 0, 0, 0 # Red, green and blue sum

      for ky in range(h):
        for kx in range(w):
          lx = kx - cw # Locale x, also called i in equation 4
          ly = ky - ch # Locale y, also called j in equation 4

          # Skip bounderies which makes them darker
          #if 0 < x - lx < imgWidth and 0 < y - ly < imgHeight:
          #  r, g, b = inputImage[y - ly][x - lx][0:3] * kernel[ky][kx]
          #  sr += r; sg += g; sb += b

          # Make use of the closest available pixel from beyond the edges
          r = inputImage[max(0, min(imgHeight - 1, y - ly))][max(0, min(imgWidth - 1, x - lx))] * kernel[ky][kx]
          sr += r

      outputImage[y][x] = sr

  return outputImage


# Shortcut for adding a single filter to an image and returning it with the correct type
def applyFilter(inputImage, kernel):
  return spatialConvolution(inputImage, kernel)


# Normalizes the array making the sum of all values equal to 1
def normalizeKernel(kernel):
  return np.multiply(kernel, 1 / np.sum(kernel))


# Our better grey function from task 2
def grey(image):
	img = image
	img = np.array(img)
	r = img[..., 0]
	g = img[..., 1]
	b = img[..., 2]

	avgGrey = (r * 0.2126 + g * 0.7152 + b * 0.0722)

	img[..., 0] = avgGrey
	img[..., 1] = avgGrey
	img[..., 2] = avgGrey
	return image


# List of useful filters. They should be described by their names. They are also normalized if necessary.
gauss3x3 = [
  [1, 2, 1],
  [2, 4, 2],
  [1, 2, 1]
]
gauss3x3 = normalizeKernel(gauss3x3)

gauss5x5 = [
  [1, 4, 6, 4, 1],
  [4, 16, 24, 16, 4],
  [6, 24, 36, 24, 6],
  [4, 16, 24, 16, 4],
  [1, 4, 6, 4, 1]
]
gauss5x5 = normalizeKernel(gauss5x5)

blur3x3 = [
  [1, 1, 1],
  [1, 1, 1],
  [1, 1, 1]
]
blur3x3 = normalizeKernel(blur3x3)

handshake = [
  [1, 0, 0, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1]
]
handshake = normalizeKernel(handshake)

sharpen = [
  [0, -2, 0],
  [-2, 9, -2],
  [0, -2, 0]
]

edges = [
  [0, -2, 0],
  [-2, 8, -2],
  [0, -2, 0]
]

# Filter for shifting all pixels to the right. Remember that the kernel is mirrored in both x and y axis when applied
rightshift = [
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0]
]

# Filter for showing the magnitude of gradient vertically
sobelX = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
]

# Filter for showing the magnitude of gradient horizontally
sobelY = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
]

image = image / 255
# Create a grey scaled image
greyImage = grey(image)

def normalize(inputImage):
  minimum = np.min(inputImage)
  inputImage = inputImage - minimum
  maximum = np.max(inputImage)
  return inputImage / maximum

# Generate a image with the sobel filter
# Notice the absolute conversion. That is why the background is black instead of an average grey
imgX = normalize((spatialConvolution(greyImage, sobelX)))
imgY = normalize((spatialConvolution(greyImage, sobelY)))

# Adding the sobel filters together
img = normalize(np.sqrt(np.power(imgX, 2) + np.power(imgY, 2)))


_, ax = plt.subplots(1, 3, figsize=(30, 16))

# Original image
ax[0].imshow(imgX, cmap='gray')
ax[0].set_axis_off()

# Display a smoothed image with gauss on out grey scaled image
ax[1].imshow(imgY, cmap='gray')
ax[1].set_axis_off()

# Display the image with the absolute of both sobel filters added together
ax[2].imshow(img, cmap='gray')
ax[2].set_axis_off()

plt.show()