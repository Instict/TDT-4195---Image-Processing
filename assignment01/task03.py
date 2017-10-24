import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 
import matplotlib.image as mpimg
from scipy import misc
from PIL import Image

#	intensity transformations
image = misc.imread('./images/4.2.06-lake.tiff')
plt.imshow(image)
plt.axis('off')
plt.show()