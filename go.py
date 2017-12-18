import matplotlib.pyplot as plt
import numpy as np
#D = 1e-7
rho = 1.225
nu = 1.48e-5
#u0 = 10.
D = 1
#nu = 1e-2
u0 = 10
x = np.linspace(0, 5000000, 50)
y = np.arange(-5, 5, .1)
X, Y = np.meshgrid(x, y)
u = D / (rho * ( 4 * np.pi * u0 * nu) ** .5) * (X ** -.5) * np.exp(- 1 * u0 * Y ** 2 / 4 / nu / X)
#plt.xlim(0, 1)
plt.contourf(X, Y, u0 - u, 1000)
plt.show()
