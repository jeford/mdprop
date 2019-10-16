#!/usr/bin/env python
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop
from mdprop.units import K_TO_AU

from dw import DoubleWell

# Set parameters
a = 0.5
b = 1.0
kT = 1.0
tau = 1.0
damptime = 1.0
chain_len = 2
L = 4
nparticle = 5000
nc = 5
dim = 1
mass = 1.0
k = 1.0
r0 = 0.0
dt = 0.05
nstep = 5000
sim_time = dt * nstep

# Initialize particles in 1D
np.random.seed(1337)
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = mass * np.ones((nparticle,))
symbols = ['X']*nparticle

# Construct harmonic oscillator potential
pot = DoubleWell(a, b)

# Construct integrator
integ = mdprop.integrator.SIN(pot, masses, kT, L=L, tau=tau, damptime=damptime, nc=nc, dim=dim, mass_weight=False)
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

# Run it!
traj = mdprop.trajectory.Trajectory(integ)
traj.hooks[0].params['boltzmann_constant'] = 1.0 # Make standard temperature printing assume kB = 1.0
traj.append_printkeys(["doublewell_energy"])
traj.del_printkey("potential_energy")
traj.run(dt, sim_time, state)
