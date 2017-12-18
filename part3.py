import matplotlib.pyplot as plt
import numpy as np
D = 100
rho = 1.225
nu = 15.11e-6
u0 = 8.
x = np.linspace(0., 1e9, 200)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
xorigin = -8e7
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * ((X - xorigin) ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / (X - xorigin))
plt.contourf(X, Y, u0 - u, 100)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
