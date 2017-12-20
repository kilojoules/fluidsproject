import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
D = 4e3
rho = 1.225
nu = 15.11e-6
u0 = 8.
x = np.linspace(0., 1e8, 200)
y = np.arange(-100, 100, 1)
X, Y = np.meshgrid(x, y)
xorigin = -8e8
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * ((X - xorigin) ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / (X - xorigin))
cnt = plt.contourf(X, Y, u0 - u, 20, cmap=plt.cm.coolwarm)
for c in cnt.collections:
    c.set_edgecolor("face")
cb = plt.colorbar()
tick_locator = ticker.MaxNLocator(nbins=8)
cb.locator = tick_locator
cb.update_ticks()

plt.savefig('boundarywake.pdf')
plt.clf()

x = np.linspace(0., 2e10, 200)
y = np.arange(-400, 400, 1)
X, Y = np.meshgrid(x, y)
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * ((X - xorigin) ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / (X - xorigin))
f, ax = plt.subplots()
ax.plot(X[Y==0], u[Y==0] / u0, label=r'$\frac{u_{deficit}}{u_0}$', linewidth=4)
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
#plt.plot(x, 1/np.sqrt(x), label=r'$\frac{1}{\sqrt{x}}$', linewidth=4)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('x')
#plt.yscale('log')
#plt.ylabel(r'$\frac{u_0}{u_0}$', size=24)
ax.legend(prop={'size':20})
plt.tight_layout()
plt.savefig('p3centerline.pdf')
plt.clf()

x = np.linspace(0., 2e9, 200)
y = np.arange(-400, 400, 1)
X, Y = np.meshgrid(x, y)
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * ((X - xorigin) ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / (X - xorigin))
phi = u * 0
for ii in range(len(u[:, 1])):
   for jj in range(len(phi[ii, :])):
      phi[ii, jj] = float(jj) / len(phi[ii, :]) * u[ii, jj] + phi[ii,jj-1]
plt.contour(X, Y, phi, 40, colors='k')
#plt.contour(X, Y, phi, levels=[0, .00001, .01, .1, 1, 10, 50, 100, 1000, 1e5, 1e10, 1e50, 1e100, 1e150, 1e200, 1e500], colors='k')
plt.savefig('p3streams.pdf')
plt.clf()

x = np.linspace(0., 1e9, 200)
y = np.arange(-100, 100, 1)
X, Y = np.meshgrid(x, y)
xorigin = -8e8
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * ((X - xorigin) ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / (X - xorigin))
cnt = plt.contourf(X, Y, u, 100, cmap=plt.cm.coolwarm)
for c in cnt.collections:
    c.set_edgecolor("face")
plt.colorbar()
plt.savefig('boundary2.pdf')
plt.clf()
