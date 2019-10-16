#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop

# Set parameters
kT = 1.0
nparticle = 1000
dim = 1
k = 1.0
r0 = 0.0
dt = 0.01
nstep = 1000
sim_time = dt * nstep

# Initialize particles in 1D
np.random.seed(1337)
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(kT, masses, dim)
V = mdprop.init.rescale_velocity(V, masses, kT)
symbols = ['X']*nparticle

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'symbols': symbols,
        }

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(r0, k)

# Construct updates
vel_update = mdprop.update.IsokineticVelocityUpdate(bound, masses, kT)

# Construct integrator
integ = mdprop.integrator.VelocityVerlet(vel_update)
print(integ)

# Run it!
traj = mdprop.trajectory.Trajectory(integ)
traj.hooks[0].params['boltzmann_constant'] = 1.0 # Make temperature printing assume kB = 1.0
traj.run(dt, sim_time, state)
