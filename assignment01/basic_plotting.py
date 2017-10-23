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

xs = np.linspace(0, 4*np.pi, 100)  # 100 points in the interval [0 and 4*pi]
ys_sin = np.sin(xs)
ys_cos = np.cos(xs)

plt.figure()
plt.plot(xs, ys_sin, label='Sine')
plt.plot(xs, ys_cos, label='Cosine')
plt.xlim(-1, 4*np.pi)  # Limit what is displayed in the x direction
plt.legend()  # Requires that each curve has a label
plt.grid(linestyle='dashed')
plt.show()

xs = np.linspace(-4, 4, 100)  # 100 points between -4 and 4
sigma = 1.0
mu = 0.0
ys = np.exp(-(xs - mu)**2 / (2 * sigma**2))
ys /= np.sqrt(2 * np.pi * sigma**2)
plt.figure()
plt.plot(xs, ys, linestyle='dashed', color='black')
plt.show()

normal = lambda xs, mu, sigma: np.exp(-(xs - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
xs = np.linspace(-5, 5, 100)
plt.figure()
plt.plot(xs, normal(xs, 0.0, 1.0), linestyle='solid',
         color='black', label=r'$\mu=0.0,\quad\sigma=1.0$')
plt.plot(xs, normal(xs, 0.0, 1.5), linestyle='dotted',
         color='black', label=r'$\mu=0.0,\quad\sigma=1.5$')
plt.plot(xs, normal(xs, 2.0, 0.5), linestyle='dashed',
         color='black', label=r'$\mu=2.0,\quad\sigma=0.5$')
plt.xticks(np.arange(-5, 5+1))
plt.legend()
plt.show()