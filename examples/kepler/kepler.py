#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop
import mdprop.wrapper
from mdprop.units import KCAL_MOL_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, AMU_TO_AU

pot = mdprop.potential.Kepler(1.0)
masses = np.array([1.0])
X, V = pot.init_cond(0.8)
symbols = ['X']

state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
    'potential_energy': pot.compute_energy(X)
}
print(X, V)

# Set up parameters for dynamics
dt = 0.005
sim_time = 10.0
print(sim_time / dt)

# Construct update objects using pointers to different forces
vel_update = mdprop.update.VelocityUpdate(pot.compute_forces)

# Construct integrator
integ = mdprop.integrator.VelocityVerlet(vel_update)
print(integ)

# Save energy, times to h5
h5hook = mdprop.hook.WriteH5("vv.h5", ['total_energy', 'simulation_time'], [(1, ), (1, )], ['f', 'f'], cache_size=1)

traj = mdprop.trajectory.Trajectory(integ)
traj.append_hook(h5hook)
traj.run(dt, sim_time, state)

TE = h5hook.h5file['total_energy'][...]
Ts = h5hook.h5file['simulation_time'][...]
plt.figure()
plt.plot(Ts, TE)
plt.ylim([-0.5001, -0.497])
plt.xlabel('Simulation Time')
plt.ylabel('Energy')
plt.title("Total Energy vs. Time")
plt.tight_layout()
plt.savefig("vv.eps")
