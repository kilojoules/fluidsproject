import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
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
xopt = [998.420161172,997.892457135,-4.74481196185e-06,832.939920769,584.16626213,978.18044547,999.644420032,842.589538267,1.34591206614e-05,133.623065854,730.445219195,544.725232405,221.477631494,838.82675197,522.785110288,360.30146187,835.144597417,633.898497028,989.925791454,161.279221469,760.330720536,193.69185915,361.690414286,416.438352986,118.745878242,1.20372267577,960.061259237,94.1089789235,114.873288645,741.673372272,263.601298579,332.881179399,888.712314314,597.132198722,963.822309229,809.161568849,834.540329659,678.135365069,482.455236641,1.96041701718]
#xopt = [996.813135827,774.460057714,231.855933068,900.511319707,961.654474656,994.364817647,925.737793192,777.867069715,886.903271222,836.963423175,413.227166103,506.542177231,48.467637848,12.4446594112,971.386060445,171.118989808,741.817056042,987.894633756,303.843613385,597.155484923,726.357252383,683.403852426,985.19644847,358.282987406,416.461430996,960.119629225,200.251604452,883.169973963,274.993881529,733.384035433,598.567377285,163.647553566,903.03806929,614.841974974,645.915501097,792.954492055,517.869799162,143.160216976,3.1174209811,809.358571317]
txopt = xopt[:nturbs]
tyopt = xopt[nturbs:]
plt.contourf(X, Y, uwake(x, np.atleast_2d(y).T, txopt, tyopt), 100)
c = plt.colorbar()
c.set_label('Wind Speed (m/s)')
plt.scatter(txopt, tyopt, c = power(uwake(tx,ty, tyopt, tyopt)) / 1e6, cmap=plt.cm.coolwarm)
plt.title("Power = %f MW" % ( np.sum(power(uwake(txopt, tyopt, txopt, tyopt)))/1e6))
plt.savefig('optlayout1.pdf')
plt.clf()
