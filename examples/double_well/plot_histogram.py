import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop

from dw import DoubleWell

nskip = 200
kT = 1.0
a = 0.5
b = 1.0
nbins = 50
nframes_corr = 10 # How many frames until they are no longer correlated

pot = DoubleWell(a, b)

Xs, _ = mdprop.io.read_traj_xyz("coors.xyz")
Xs = Xs[nskip::nframes_corr]
print(np.shape(Xs))

xmax = np.max(np.abs(Xs)) + 0.1
xmin = -xmax
xs = np.linspace(xmin, xmax)
Es = pot.compute_energy_per_particle(xs)
ys = np.exp(-Es / kT)
ys /= np.sum(ys * (xs[1]-xs[0]))

plt.figure()
plt.plot(xs, ys)
plt.hist(np.ravel(Xs), bins=nbins, normed=True)
plt.savefig("histogram.pdf")
