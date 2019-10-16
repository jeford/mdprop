import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop

pot = mdprop.potential.Prinz()
X = np.array([[-1.0]])
V = np.array([[ 0.0]])
masses = np.array([1.0])
symbols = ['X']
beta = 0.1
dt = 0.01
sim_time = 10.0

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        'potential_energy': pot.compute_energy(X),
    }

vup = mdprop.update.VelocityUpdate(pot.compute_force)

integ = mdprop.integrator.VelocityVerlet(vup)

traj = mdprop.trajectory.Trajectory(integ)

gradhook = mdprop.hook.Gradient(vup)
tschook = mdprop.hook.TimestepController(mdprop.control.exponential, beta)
h5hook = mdprop.hook.WriteH5("prinz.h5", ["X", "potential_energy", "total_energy", "simulation_time"], [(1, 1), (1, ), (1, ), (1, )], ['f', 'f', 'f', 'f'], cache_size=1)
traj.append_hooks([gradhook, h5hook, tschook])
traj.append_printkey('dt')
traj.run(dt, sim_time, state)

h5file = h5hook.h5file
Xs = h5file['X'][:, 0, 0]
Ts = h5file['simulation_time'][:, 0]
PEs = h5file['potential_energy'][:, 0]
TEs = h5file['total_energy'][:, 0]
dts = Ts[1:] - Ts[:-1]
dts_scaled = (dts - min(dts))/max(dts - min(dts)) * 4.0
#dts_scaled = dts / min(dts) * 2.0

plt.figure()
plt.plot(Xs, PEs)
plt.plot(Xs[1:], dts_scaled)
plt.savefig("prinz.eps")

plt.figure()
plt.plot(Ts, TEs)
plt.savefig("energy.eps")
