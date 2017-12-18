import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
U = 1.
m = 3.
BOUNDS = 400.
x, y = np.linspace(-8 * BOUNDS, BOUNDS, 100), np.linspace(-1 * BOUNDS, BOUNDS, 100)
#x, y = np.linspace(-1 * BOUNDS, BOUNDS, 100), np.linspace(-1 * BOUNDS, BOUNDS, 100)
def phi(x,y):
   ii = 1
   mysumm = 0.
   for dell in np.arange(-1., 1., .05):
      ii += 1
      mysumm += (-1 ** ii) * m * np.arctan2((y + dell), x) / 2 / np.pi
   return U*y + mysumm
X, Y = np.meshgrid(x,y)
print x
print phi(X, Y)
plt.contour(x, y, phi(X, Y), 20, colors='k')
plt.show()

