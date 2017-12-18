from scipy.optimize import minimize as mini
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
k = 0.075
rho = 1.225
uinf = 8.
D = 80.
nturbs = 20
a = np.ones(nturbs) * 0.333333
spacing = 4
nx = 5 * spacing
xpoints = np.linspace(start=spacing*D, stop=4*spacing*D, num=4)
ypoints = np.linspace(start=spacing*D, stop=5*spacing*D, num=5)
xpoints, ypoints = np.meshgrid(xpoints, ypoints)
tx = np.ndarray.flatten(xpoints)
ty = np.ndarray.flatten(ypoints)
x = np.arange(0, 4 * spacing * D, 1)
y = np.arange(0, 5 * spacing * D, 1)
X, Y = np.meshgrid(x, y)
nx, ny = len(x), len(y)
#X, Y = np.meshgrid(np.linspace(0, 4 * spacing*D), np.linspace(0, 5 * spacing*D))

uinf = 10.

def C(x, t):
   return ( D / (D + 2 * k * (t - x)*dist)) ** 2

def uwake(a, tx, ty):
   udef = np.zeros((nturbs, nx, nx))
   u = np.ones((nturbs, nx, ny))
   tx, ty = zip(*sorted(zip(tx, ty), key=itemgetter(0)))
   for turb in range(nturbs):
      y_idx = np.logical_and(ty[turb] - D/2 - k * x < np.atleast_2d(y).T, np.atleast_2d(y).T < ty[turb] + D/2 + k*x)
      u0 = u[turb][np.where(x==int(tx[turb]))[0], np.where(y==int(ty[turb]))[0]]
      s =  2 * a[turb] * u0 / ( 1 + 2 * k * (x - x[int(tx[turb])]))
      udef[turb, x>tx[turb], :][y_idx.T] += s
      u = u - udef
   return u, udef
u, ud = uwake(a, tx, ty)
plt.imshow(u) ; plt.show() ; quit()

def uwake(a):
  u = np.zeros(nturbs)
  for turb in range(nturbs):
    for t in range(turb):
      if turb == 0: pass
      else: u[turb] += (a[turb-1] * C(t, turb)) ** 2
      #print (a[turb] * C(t, turb)) ** 2,
    u[turb] =  uinf * (1 - 2 * np.sqrt(u[turb]))
    #print turb, a[turb], u[turb], .5 * rho * u[turb]**3 * D ** 2 / 4 * np.pi / 1e3
  return u

def power(a): return -1 * np.sum( 4 * a * (1 - a)**2 * .5 * rho * uwake(a)**3 * D ** 2 / 4 * np.pi / 1e3)
