#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import KCAL_MOL_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, AMU_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/nitromethane20.xyz"  # XYZ file of geometry
kT0 = 600.0 * K_TO_AU                    # Initial temperature sampling for Boltzmann 
dt = 1.0 * FS_TO_AU                     # Timestep 
nstep = 5000                            # Number of steps to take in MD
sim_time = nstep * dt                   # Time to take in MD
kT = 1800.0 * K_TO_AU                   # Thermostat temp
damptime = 10.0 * FS_TO_AU               # Damping parameter for Langevin thermostat (fs)
density = 1.14                          # Density to calculate radius of spherical piston (g/cm^3)
rscale = 0.9                            # Scaling factor of radius to increase pressure

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
bxde = mdprop.update.BXDE(masses, adaptive=True) # Defaults to adaptive scheme that only reflects from below in energy

boundary_r = mdprop.utils.density_to_spherical_radius(masses, density) * rscale
boundary_k = np.reshape(masses, (-1, 1)) * 10.0 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2 / AMU_TO_AU # a.u. = 10.0 kcal/mol/Angstrom**2/amu
boundary_pot = mdprop.potential.SoftSphere(radius=boundary_r, magnitude=boundary_k, mass_dependent=False)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    pot_list = mdprop.potential.PotentialList([DFTB, boundary_pot])
    vel_update = mdprop.update.VelocityUpdate(pot_list, masses)
    state.update(vel_update.update(0.0, state)) # Update initial state to include energy/gradient data

    # Construct integrator (Langevin / BXDE / DFTB / Boundary / Position / Boundary / DFTB / BXDE / Langevin)
    integ = mdprop.integrator.VelocityVerletMultiple([thermo, bxde, vel_update])
    print(integ)

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    traj.append_printkey("boundary_energy")
    traj.append_printkey("BXDE_PE_min")
    traj.append_printkey("BXDE_PE_curr_max")
    traj.run(dt, sim_time, state)
