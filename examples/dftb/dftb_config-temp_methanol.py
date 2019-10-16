#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import FS_TO_AU, K_TO_AU

## PARAMETERS ##
np.random.seed(1337)
xyzfile = "../data/dialanine.xyz"   # XYZ file of geometry
kT = 600.0 * K_TO_AU              # Temperature sampling for Boltzmann 
dt = 1.0 * FS_TO_AU                # Timestep 
nstep = 100                        # Number of steps to take in MD
sim_time = nstep * dt              # Time to take in MD
damptime = 5.0 * FS_TO_AU          # Damping parameter for Langevin thermostat (fs)

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
V = mdprop.init.boltzmann(kT, masses)
X, V = mdprop.init.initialize_centered(X, V, masses)
state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
}

# Construct Langevin thermostat
thermo = mdprop.update.Langevin(masses, kT, damptime, rescale=True)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    vel_update = mdprop.update.VelocityUpdate(DFTB, masses)

    # Construct integrator (Langevin / DFTB / Position)
    integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
    print(integ)

    conf_temp_hook = mdprop.hook.ConfigurationalTemperature(DFTB)

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    traj.append_hook(conf_temp_hook)
    #traj.append_printkey("config_temperature")
    traj.append_printkey("config1_temperature")
    traj.append_printkey("config2_temperature")
    traj.append_printkey("config3_temperature")
    traj.run(dt, sim_time, state)
