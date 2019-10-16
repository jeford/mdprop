#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import KCAL_MOL_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, AMU_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/nitromethane20.xyz"  # XYZ file of geometry
dt = 2.0 * FS_TO_AU                     # Timestep 
nstep = 50                              # Number of steps to take in MD
sim_time = nstep * dt                   # Time to take in MD
kT = 600.0 * K_TO_AU                    # Thermostat temp
density = 1.14                          # Density to calculate radius of spherical piston (g/cm^3)
rscale = 0.9                            # Scaling factor of radius to increase pressure

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
V = mdprop.init.boltzmann(kT, masses)
X, V = mdprop.init.initialize_centered(X, V, masses)
V = mdprop.init.rescale_velocity(V, masses, kT)

state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
}

boundary_r = mdprop.utils.density_to_spherical_radius(masses, density) * rscale
boundary_k = np.reshape(masses, (-1, 1)) * 10.0 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2 / AMU_TO_AU # a.u. = 10.0 kcal/mol/Angstrom**2/amu
boundary_pot = mdprop.potential.SoftSphere(radius=boundary_r, magnitude=boundary_k)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    pot_list = mdprop.potential.PotentialList([DFTB, boundary_pot])
    vel_update = mdprop.update.IsokineticVelocityUpdate(pot_list, masses, kT)

    # Construct integrator (DFTB / Position / DFTB)
    integ = mdprop.integrator.VelocityVerlet(vel_update)
    print(integ)

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    traj.run(dt, sim_time, state)
