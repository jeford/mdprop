#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import FS_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/methanol.xyz"    # XYZ file of geometry
kT0 = 100.0 * K_TO_AU               # Initial temperature sampling for Boltzmann 
dt = 1.0 * FS_TO_AU                 # Timestep 
nstep = 50                          # Number of steps to take in MD
sim_time = nstep * dt               # Time to take in MD
kT = 600.0 * K_TO_AU                # Langevin thermostat temp (K)
damptime = 10.0 * FS_TO_AU          # Damping parameter for Langevin thermostat (fs)

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
V = mdprop.init.boltzmann(kT0, masses)
X, V = mdprop.init.initialize_centered(X, V, masses)
state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
}

# Construct Langevin thermostat and BXDE update
thermo = mdprop.update.Langevin(masses, kT, damptime)
bxde = mdprop.update.BXDE(masses, PE_min=-6.5, dPE=0.1, adaptive=False) # Defaults to adaptive scheme that only reflects from below in energy

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    vel_update = mdprop.update.VelocityUpdate(DFTB, masses)
    state.update(vel_update.update(0.0, state)) # Update initial state to include energy/gradient data
    
    # Construct integrator (Langevin / BXDE / DFTB / Position / DFTB / BXDE / Langevin)
    integ = mdprop.integrator.VelocityVerletMultiple([thermo, bxde, vel_update])
    print(integ)

    # Create Trajectory object
    traj = mdprop.trajectory.Trajectory(integ)
    # Include prints keys for adaptive BXDE
    #traj.append_printkey("BXDE_PE_min")
    #traj.append_printkey("BXDE_PE_curr_max")
    # Run it!
    traj.run(dt, sim_time, state)
