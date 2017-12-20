import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
rho = 1.225
D = 80.
u0 = 10.
nturbs = 20
spacing = 2
k = 0.075
xpoints = np.linspace(start=spacing*D, stop=4*spacing*D, num=4)
ypoints = np.linspace(start=spacing*D, stop=5*spacing*D, num=5)
xpoints, ypoints = np.meshgrid(xpoints, ypoints)
tx = np.ndarray.flatten(xpoints)
ty = np.ndarray.flatten(ypoints)
x = np.linspace(0, 1e3, 200)
y = np.linspace(0, 1e3, 200)
a = np.ones(nturbs) * 1 / 3.
def uwake(x, y, tx, ty):
    tx, ty = zip(*sorted(zip(tx, ty), key=itemgetter(0)))
    udef = 0.
    for turb in range(nturbs):
        if tx[turb] < x:
            if (ty[turb] + D / 2 + k * (x - tx[turb]) > y and
                ty[turb] - D / 2 - k * (x - tx[turb]) < y):
                udef += 2 * a[turb] * u0 / (1 + 2 * k * (x - tx[turb]) / D) ** 2
    return u0 - udef
X, Y = np.meshgrid(x, y)
uwake = np.vectorize(uwake, excluded=(2,3,))
def power(u): return 4 * a * (1 - a) ** 2  * rho * .5 * u ** 3 * np.pi * D ** 2 / 4
plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, tx, ty), 100)
plt.colorbar()
plt.scatter(tx, ty, c = power(uwake(tx,ty, tx, ty)) / 1e6, cmap=plt.cm.coolwarm)
plt.title("Power = %f MW" % ( np.sum(power(uwake(tx,ty, tx, ty)))/1e6))
plt.savefig('grid.pdf')
plt.clf()

tx = 500 + 500 * np.sin(np.linspace(0, 2 * np.pi, 20))
ty = 500 + 500 * np.cos(np.linspace(0, 2 * np.pi, 20))
plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, tx, ty), 100)
plt.colorbar()
plt.scatter(tx, ty, c = power(uwake(tx,ty, tx, ty)) / 1e6, cmap=plt.cm.coolwarm)
plt.title("Power = %f MW" % ( np.sum(power(uwake(tx,ty, tx, ty)))/1e6))
plt.savefig('circle.pdf')
plt.clf()

tx[:] = 0
tx[10:] = 1000
ty[:10] = np.linspace(0, 1e3, 10)
ty[10:] = np.linspace(0, 1e3, 10)
plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, tx, ty), 100)
plt.colorbar()
plt.scatter(tx, ty, c = power(uwake(tx,ty, tx, ty)) / 1e6, cmap=plt.cm.coolwarm)
plt.title("Power = %f MW" % ( np.sum(power(uwake(tx,ty, tx, ty)))/1e6))
plt.savefig('edges.pdf')
plt.clf()

from scipy.optimize import minimize as mini
def fitness(x):
    ttx = x[:nturbs]
    tty = x[nturbs:]
    powa =  np.sum(power(uwake(ttx, tty, ttx, tty))) / 1e6
    return -1 * powa + spacing(x) + 1e2 * bounds(x)
def spacing(x):
    xp = x[:nturbs]
    yp = x[nturbs:]
    penalty = 0.
    for n in range(len(xp)):
      for s in range(n+1, len(xp)):
         con = np.sqrt((xp[n] - xp[s]) **2 + (yp[s] - yp[n])**2)
         if con < (1 * D): penalty += 1e2 * (2 * D - con) ** 2
    return 1 * penalty
def bounds(x):
   penalty = 0.
   for e in x:
      if e < 0: penalty += 10 * (0-e)
      if e > 1e3: penalty -= 10 * (1e3-e)
   return 1 * penalty
x0 = np.concatenate((tx, ty))
#s = np.array(list(tx) + list(ty))
#print bounds(s), spacing(s) ; quit()

confs = []
for i in range(nturbs):
    exec("confs.append(lambda x:  10000 * (0 - x[{}]) if 0 - x[{}]>0 else 0)".format(i,i))
    exec("confs.append(lambda x: -10000 * (1e3 - x[{}]) if 1e3 - x[{}]<0 else 0)".format(i,i))
    for j in range(i + 1, nturbs):
        exec("confs.append(lambda x: -10000 * np.sqrt((x[:nturbs][{}] - x[:nturbs][{}])**2 + (x[nturbs:][{}] - x[nturbs:][{}])**2) if (np.sqrt((x[:nturbs][{}] - x[:nturbs][{}])**2 + (x[nturbs:][{}] - x[nturbs:][{}])**2) < 2 * D) else 0)".format(i, j , i, j, i, j, i, j))
cons = tuple([{'type': 'ineq', 'fun': confs[i]} for i in range(len(confs))])
#cons = cons + ({'type': 'ineq', 'fun': bounds},)
ii = 0
best = 0
while True:
   miniout = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints =cons, method="COBYLA", options={'rhobeg':200, 'maxiter':20000})
   #miniout = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints =cons, method="COBYLA", options={'rhobeg':400})
   xopt = miniout.x
   #xopt = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints =cons, method="COBYLA", options={'maxiter':200000}).x
   #xopt = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints =cons, method="SLSQP", options={'maxiter':200000}).x
   #xopt = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints = ({'type': 'eq', 'fun': spacing}), method="SLSQP", options={'maxiter':200000}).x
   #xopt = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints = ({'type': 'ineq', 'fun': spacing}), method="TNC").x
#xopt = mini(fitness, x0, bounds=[[0, 1e3] for _ in range(2 * nturbs)], constraints = ({'type': 'ineq', 'fun': spacing}, {'type': 'ineq', 'fun': bounds}), method="COBYLA").x
   print spacing(xopt), bounds(xopt), ','.join([str(s) for s in xopt])
   print '--'
#   txopt = xopt[:nturbs]
#   tyopt = xopt[nturbs:]
#   plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, txopt, tyopt), 100)
#   c = plt.scatter(txopt, tyopt, c = power(uwake(tx,ty, tyopt, tyopt)) / 1e6, cmap=plt.cm.coolwarm)
#   plt.colorbar(c)
#   plt.title('Power %f'%fitness(xopt))
#   plt.show()
   if abs(bounds(xopt)) < 5 and abs(spacing(xopt)) < 1: 
       if  miniout.fun < best: 
          best = miniout.fun
          bestx = xopt
          ii += 1
       if ii == 10: break
       print ii, ' / 10     <<<<----'
       print 'BEST IS ', best, '(', miniout.fun,')'
       x0 = bestx
   if spacing(xopt) < spacing(x0): 
      x0 = xopt
   else: 
      x0 = np.random.uniform(0, 1e3, nturbs * 2)
xopt = bestx
txopt = xopt[:nturbs]
tyopt = xopt[nturbs:]

plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, txopt, tyopt), 100)
c = plt.scatter(txopt, tyopt, c = power(uwake(tx,ty, tyopt, tyopt)) / 1e6, cmap=plt.cm.coolwarm)
plt.colorbar(c)
plt.title("Power = %f MW" % ( np.sum(power(uwake(txopt, tyopt, txopt, tyopt)))/1e6))
plt.savefig('optlayout.pdf')
plt.clf()
quit()

plt.scatter(txopt, tyopt) ; plt.show()
