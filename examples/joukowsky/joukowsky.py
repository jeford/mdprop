import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop

class Joukowsky(mdprop.potential.Potential):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute_energy_per_particle(self, X, **state):
        r = np.linalg.norm(X, axis=1)
        return self.alpha * r + 1.0/r

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        r = np.linalg.norm(X, axis=1)[:, None]
        V = np.sum(self.alpha * r + 1.0 / r)
        G = self.alpha * X / r - X / (r ** 3)
        return V, G

# Params
dt = 0.05
sim_time = 13.0
beta = 0.2

x = np.linspace(0.1, 2.0, 100)[:, None]
pot = Joukowsky(5.0)
y = pot.compute_energy_per_particle(x)
_, g = pot.compute_gradient(x)
ng = mdprop.utils.numerical_gradient(x, pot.compute_energy)
print(max(g - ng))

vup = mdprop.update.VelocityUpdate(pot.compute_force)
integ = mdprop.integrator.VelocityVerlet(vup)

X = np.array([[-2.0]])
V = np.array([[0.0]])
masses = np.array([1.0])
symbols = ['X']
init_pot = pot.compute_energy(X)

init_state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        'potential_energy': init_pot,
    }

state = init_state.copy()
traj = mdprop.trajectory.Trajectory(integ)
h5hook = mdprop.hook.WriteH5('joukowsky.h5', ['X', 'total_energy'], [(1, 1), (1, )], ['f', 'f'], cache_size=1)
traj.append_hook(h5hook)
traj.run(dt, sim_time, state)

xs = h5hook.h5file['X'][:, :, 0]
es = h5hook.h5file['total_energy'][:, 0]
vs = pot.compute_energy_per_particle(xs)

state = init_state.copy()
traj = mdprop.trajectory.Trajectory(integ)
gradhook = mdprop.hook.Gradient(vup)
h5hook_vts = mdprop.hook.WriteH5('joukowsky_vts.h5', ['X', 'total_energy'], [(1, 1), (1, )], ['f', 'f'], cache_size=1)
tshook = mdprop.hook.TimestepController(mdprop.control.exponential, beta)
traj.append_hooks([gradhook, h5hook_vts, tshook])
traj.run(dt, sim_time, state)

xs_vts = h5hook_vts.h5file['X'][:, :, 0]
es_vts = h5hook_vts.h5file['total_energy'][:, 0]
vs_vts = pot.compute_energy_per_particle(xs_vts)

print(len(xs))
print(len(xs_vts))

# Plot the VV trajectory
xmin = np.min(xs)
xmax = np.max(xs)
x = np.linspace(xmin, xmax, 100)[:, None]
y = pot.compute_energy_per_particle(x)

plt.figure()
plt.plot(x, y)
plt.plot(xs, vs, marker='o', ls='', label='Potential')
plt.plot(xs, es, marker='o', color='g', label='Total')
plt.xlabel("X")
plt.ylabel("Energy")
plt.legend()
plt.tight_layout()
plt.savefig("joukowsky.eps")

# Plot the VV VTS trajectory
xmin_vts = np.min(xs_vts)
xmax_vts = np.max(xs_vts)
x_vts = np.linspace(xmin_vts, xmax_vts, 100)[:, None]
y_vts = pot.compute_energy_per_particle(x_vts)

plt.figure()
plt.plot(x_vts, y_vts)
plt.plot(xs_vts, vs_vts, marker='o', ls='', label='Potential')
plt.plot(xs_vts, es_vts, marker='o', color='g', label='Total')
plt.xlabel("X")
plt.ylabel("Energy")
plt.legend()
plt.tight_layout()
plt.savefig("joukowsky_vts.eps")
