import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy.abc import x, y
BOUNDS = 400.
downstream = 8
m = 3.
def stream_function(U=8, R=1):
    #r = sympy.sqrt(x**2 + y**2)
   ii = 1
   mysumm = 0.
   for dell in np.arange(-20., 20., .5):
      ii += 1
      mysumm += (-1 ** ii) * m * sympy.atan2((y + dell), x) / 2 / np.pi
      mysumm += (-1 ** ii) * m * sympy.atan2((y - dell), x + BOUNDS * downstream * .5) / 2 / np.pi
   return U*y + mysumm

def velocity_field(psi):
    u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
    v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
    return u, v

def plot_streamlines(ax, u, v, psi, xlim=(-1, 1), ylim=(-1, 1)):
    x0, x1 = xlim
    y0, y1 = ylim
    #Y, X =  np.ogrid[y0:y1:100j, x0:x1:100j]
    X, Y = np.meshgrid(np.linspace(-downstream * 1.5 * BOUNDS, BOUNDS, 100), np.linspace(-1 * BOUNDS, BOUNDS, 100))
    ax.streamplot(X, Y, u(X, Y), v(X, Y), color='cornflowerblue')

psi = stream_function()
u, v = velocity_field(psi)

xlim = ylim = (-3, 3)
fig, ax = plt.subplots(figsize=(4, 4))
plot_streamlines(ax, u, v, psi, xlim, ylim)

plt.show()
