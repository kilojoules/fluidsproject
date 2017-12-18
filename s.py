import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1e7, 100)
plt.plot(x, 1 / np.sqrt(x), label=r'$\frac{1}{\sqrt{x}}$')
plt.plot(x, np.exp(-1/x) / np.sqrt(x), label=r'$\frac{e^{\frac{-1}{x}}}{\sqrt{x}}$')
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('x', size=20)
plt.legend()
plt.show()
