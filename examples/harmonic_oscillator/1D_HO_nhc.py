#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop
from mdprop.units import K_TO_AU

# Set parameters
kT = 1.0
damping_time = 1.0
chain_len = 5
nparticle = 1000
dim = 1
k = 1.0
r0 = 0.0
dt = 0.05
nstep = 1000
sim_time = dt * nstep

# Initialize particles in 1D
np.random.seed(1337)
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(kT, masses, dim)
symbols = ['X']*nparticle
aux_masses = np.ones((chain_len, nparticle)) * kT * damping_time**2
aux_q, aux_v = mdprop.update.NoseHooverNVT.initialize(kT, aux_masses)

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        'aux_position_NH': aux_q,
        'aux_velocity_NH': aux_v,
        }

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(r0, k)

# Construct updates
vel_update = mdprop.update.VelocityUpdate(bound, masses)
thermo = mdprop.update.NoseHooverNVT(masses, kT, aux_masses)

# Construct integrator
integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
print(integ)

# Run it!
traj = mdprop.trajectory.Trajectory(integ)
traj.append_printkey("NHC_energy")
traj.hooks[0].params['boltzmann_constant'] = 1.0 # Make temperature printing assume kB = 1.0
traj.run(dt, sim_time, state)
