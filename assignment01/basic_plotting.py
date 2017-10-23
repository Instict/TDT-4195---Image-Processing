import numpy as np
#	matplotlib inline
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image

xs = np.linspace(0, 5, 6)	#	6 points between 0 and 5
print('f: {} -> {}'.format(xs,xs))

plt.figure()
plt.plot(xs, xs)
plt.title('A line')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()