#!/usr/bin/env python
import numpy as np

import mdprop

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Set up parameters for dynamics
alpha = 1.5
dt = 0.005
sim_time = 10.0

# Define Hairer's time step controller
def hairer_time_step_control(state, alpha=1.5):
    return -alpha * np.sum(state['X'] * state['V']) / np.sum(state['X'] ** 2)

# Construct update objects using pointers to different forces
vel_update = mdprop.update.VelocityUpdate(pot.compute_forces)

# Construct integrator
integ = mdprop.integrator.VelocityVerlet(vel_update)
print(integ)

# Save energy, times to h5
h5hook = mdprop.hook.WriteH5("hairer.h5", ['kinetic_energy', 'total_energy', 'simulation_time'], [(1, ), (1, ), (1, )], ['f', 'f', 'f'], cache_size=1)

traj = mdprop.trajectory.Trajectory(integ)
ts_control = mdprop.hook.TimestepController(hairer_time_step_control, alpha)
traj.append_hooks([ts_control])
traj.append_hook(h5hook)
traj.append_printkeys(['dt', 'control'])
traj.run(dt, sim_time, state)

KE = h5hook.h5file['kinetic_energy'][...]
TE = h5hook.h5file['total_energy'][...]
Ts = h5hook.h5file['simulation_time'][...]
#TE = TE - TE[0]
dts = Ts[1:] - Ts[:-1]
print(len(dts))

fig, ax1 = plt.subplots()
ax1.plot(Ts, TE, color='tab:blue', ls='-', label='Total energy')
ax2 = ax1.twinx()
ax2.plot(Ts[1:], dts, color='tab:red', ls='--')
ax1.set_ylim([-0.5001, -0.497])
ax1.set_xlabel('Simulation Time')
ax1.set_ylabel("Total Energy", color='tab:blue')
ax2.set_ylabel("Time step", color='tab:red')
plt.title("Total Energy vs. Time")
plt.tight_layout()
plt.savefig("hairer_total.eps")

fig, ax1 = plt.subplots()
ax1.plot(Ts, KE, color='tab:blue', ls='-', label='Kinetic energy')
ax2 = ax1.twinx()
ax2.plot(Ts[1:], dts, color='tab:red', ls='--')
ax1.set_xlabel('Simulation Time')
ax1.set_ylabel("Kinetic Energy", color='tab:blue')
ax2.set_ylabel("Time step", color='tab:red')
plt.title("Kinetic Energy vs. Time")
plt.tight_layout()
plt.savefig("hairer_kinetic.eps")
