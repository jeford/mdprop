#!/usr/bin/env python
import sys
import numpy as np

import mdprop
from mdprop.units import K_TO_AU

# Set parameters
kT = 1.0
tau = 1.0
damptime = 1.0
chain_len = 2
L = 4
nparticle = 100
nc = 5
dim = 1
mass = 1.0
k = 1.0
r0 = 0.0
dt = 0.05
nstep = 1000
sim_time = dt * nstep

# Initialize particles in 1D
np.random.seed(1337)
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = mass * np.ones((nparticle,))
symbols = ['X']*nparticle

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(r0, k)

# Construct integrator
integ = mdprop.integrator.SIN(bound, masses, kT, L=L, tau=tau, damptime=damptime, nc=nc, dim=dim, mass_weight=False)
print(integ)

Qs = integ.updates[0].params['Qs']
V, aux_v = mdprop.init.sin_velocity(masses, Qs, kT)

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        'aux_velocity_NH': aux_v,
        }

print(np.linalg.norm(state['V']**2 * np.reshape(masses, (-1, 1))+ L / (L+1.0) * np.sum(Qs[0] * state['aux_velocity_NH'][0]**2, axis=0) - L * kT)) # First value in constraint vector
# Construct hook that tracks the value of our constraint, as well as
# configurational temperature (SIN thermostat is only canonical in configuration
# space and not phase/momentum space)
class lmbd_hook(mdprop.hook.Hook):
    def __init__(self):
        return

    def compute(self, state):
        update = {
                "constraint_violation" : np.linalg.norm(state['V']**2 * np.reshape(masses, (-1, 1))+ L / (L+1.0) * np.sum(Qs[0] * state['aux_velocity_NH'][0]**2, axis=0) - L * kT), # First value in constraint vector
            }
        G = state.get('softsphere_gradient', None)
        if G is not None:
            update["virial_temperature"] = mdprop.utils.compute_virial_temperature(state['X'], -G), # r^T F, equivalent to configurational temperature for HO
        return update

# Run it!
traj = mdprop.trajectory.Trajectory(integ)
traj.hooks[0].params['boltzmann_constant'] = 1.0 # Make standard temperature printing assume kB = 1.0
traj.hooks.append(lmbd_hook()) 
traj.append_printkeys(["virial_temperature", "constraint_violation", "softsphere_energy"])
traj.del_printkey("potential_energy")
traj.run(dt, sim_time, state)
