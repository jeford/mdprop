import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop

x = np.linspace(-0.3, 0.3, 100)[:, None]
pot = mdprop.potential.Prinz()
y = pot.compute_energy_per_particle(x)
_, g = pot.compute_gradient(x)
plt.figure()
plt.plot(x, y)
plt.plot(x, g)
plt.savefig("prinz.eps")
