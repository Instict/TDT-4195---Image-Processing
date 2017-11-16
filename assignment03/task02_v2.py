import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image




imagePath = {'yeast' : './images/Fig1043(a)(yeast_USC).tiff',
			'weld' : './images/Fig1051(a)(defective_weld).tiff',
			'noisy' : './images/noisy.tiff',
			'task5one' : './images/task5-01.tiff',
			'task5two' : './images/task5-02.tiff',
			'task5three' : './images/task5-03.tiff'}

image = misc.imread(imagePath['weld'])
image = np.array(image, dtype=np.float32)

plt.plot()
plt.axis('off')
plt.title('Original image')
plt.imhsow(image, cmap = 'gray')
plt.show()
