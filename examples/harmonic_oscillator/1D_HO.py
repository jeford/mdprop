#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop
from mdprop.units import K_TO_AU

# Set parameters
kT = 0.25  # Kelvin
nparticle = 10
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

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        }

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(r0, k)

# Construct integrator
vel_update = mdprop.update.VelocityUpdate(bound, masses)
integ = mdprop.integrator.VelocityVerlet(vel_update)
print(integ)

# Run it!
traj = mdprop.trajectory.Trajectory(integ)
traj.run(dt, sim_time, state)
