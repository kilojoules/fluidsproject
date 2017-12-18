import numpy as np
import matplotlib.pyplot as plt
k = 0.075
rho = 1.225
uinf = 8.
D = 80.
dist = 400.

def uwake(x, a):
   u = np.zeros(10)
   u += (a * C(x)) ** 2
   ufinal = uinf * (1 - 2 * np.sqrt(u))
   print x, u, ufinal
   return ufinal
#uwake = np.vectorize(uwake)
def C(x):
   return ( D / (D + 2 * k * (x * dist))) ** 2
x = np.arange(10)
a = np.ones(10) * 0.33
plt.plot(x, uwake(x, a))
plt.show()
