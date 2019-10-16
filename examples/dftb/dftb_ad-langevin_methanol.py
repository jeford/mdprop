#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import FS_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/nitromethane20.xyz"   # XYZ file of geometry
kT0 = 100.0 * K_TO_AU                      # Initial temperature sampling for Boltzmann 
dt = 1.0 * FS_TO_AU             # Timestep 
nstep = 5000                    # Number of steps to take in MD
sim_time = nstep * dt           # Time to take in MD
kT = 1000.0 * K_TO_AU            # Langevin thermostat temp (K)
damptime = 100.0 * FS_TO_AU     # Damping parameter for Langevin thermostat (fs)

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
V = mdprop.init.boltzmann(kT0, masses)
X, V = mdprop.init.initialize_centered(X, V, masses)
aux_mass = 0.5 * np.size(X) * kT * damptime**2                # Mass associated with gamma parameter of Ad-Langevin
state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
    'gamma': 2.0/damptime
}

# Construct Langevin thermostat
thermo = mdprop.update.AdaptiveLangevin(masses, kT, aux_mass)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    vel_update = mdprop.update.VelocityUpdate(DFTB, masses)

    # Construct integrator (Langevin / DFTB / Position)
    integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
    print(integ)

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    traj.append_printkey("gamma")
    traj.run(dt, sim_time, state)
