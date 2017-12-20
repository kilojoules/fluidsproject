import numpy as np
from scipy.optimize import minimize as mini
import matplotlib.pyplot as plt
k = 0.075
rho = 1.225
uinf = 8.
D = 80.
dist = 400.
nturbs = 10
def C(x, t):
   return ( D / (D + 2 * k * (t - x)*dist)) ** 2
a = np.ones(nturbs) * 1 / 3.
def uwake(a):
  u = np.zeros(nturbs)
  for turb in range(nturbs):
    for t in range(turb):
      if turb == 0: pass
      else: u[turb] += (a[t] * C(t, turb)) ** 2
      #print (a[turb] * C(t, turb)) ** 2,
    u[turb] =  uinf * (1 - 2 * np.sqrt(u[turb]))
    #print turb, a[turb], u[turb], .5 * rho * u[turb]**3 * D ** 2 / 4 * np.pi / 1e3
  return u

def power(a): 
  return -1 * np.sum( 4 * a * (1 - a) ** 2 * .5 * rho * uwake(a) ** 3 * D ** 2 / 4 * np.pi / 1e3)
f, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(range(nturbs), uwake(a), c='k', label="Wind Speed")
ax2.plot(range(nturbs), 4 * a * (1 - a ) ** 2 * .5 * rho * uwake(a)**3 * D ** 2 / 4 * np.pi / 1e3, ls='--', c='k', label="Power")
plt.xlabel('Turbine')
ax.set_ylabel('Speed')
ax2.set_ylabel('Power (kW)')
ax.legend()
ax2.legend( loc="upper right", bbox_to_anchor=[.97, .92])
plt.xlabel("Turbine Number")
plt.title("power is %f kW"%(-1 * power(a)))
plt.savefig('speed.pdf')
plt.clf()

#def con1(a): return np.sum(np.abs(a[a>.3333]))
#def con2(a): return np.sum(np.abs(a[a<0]))
#astar = mini(power, a, constraints=[{'type': 'ineq', 'fun': con1}, {'type': 'ineq', 'fun': con2}], method='COBYLA').x
astar = mini(power, a, bounds=[(0, .3333) for _ in range(nturbs)], method='COBYLA').x
f, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(range(nturbs), uwake(astar), c='k', label="Wind Speed")
ax2.plot(range(nturbs), 4 * a * (1 - a) ** 2 * .5 * uwake(astar)**3 * rho, c='k', ls='--', label='Power')
ax.get_yaxis().get_major_formatter().set_scientific(False)
ax.get_yaxis().get_major_formatter().set_useOffset(False)
ax2.get_yaxis().get_major_formatter().set_scientific(False)
ax2.get_yaxis().get_major_formatter().set_useOffset(False)

plt.draw()
plt.xlabel("Turbine Number")
plt.tight_layout()
plt.title("power is %f"%power(astar))
ax.legend()
ax2.legend( loc="upper right", bbox_to_anchor=[.97, .92])
ax2.set_ylabel("Power (kW)")
plt.savefig('optimized.pdf')
print 'baseline ', power(a), '   optimized ', power(astar), astar
print ', '.join(str(s) for s in astar)

