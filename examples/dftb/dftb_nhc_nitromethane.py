#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import KCAL_MOL_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, AMU_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/nitromethane20.xyz"  # XYZ file of geometry
kT0 = 600.0 * K_TO_AU                    # Initial temperature sampling for Boltzmann 
dt = 0.5 * FS_TO_AU                     # Timestep 
nstep = 5000                            # Number of steps to take in MD
sim_time = nstep * dt                   # Time to take in MD
kT = 600.0 * K_TO_AU                   # Thermostat temp
damptime = 10.0 * FS_TO_AU               # Damping parameter for Langevin thermostat (fs)
density = 1.14                          # Density to calculate radius of spherical piston (g/cm^3)
rscale = 0.9                            # Scaling factor of radius to increase pressure
nhc_len = 5                             # Number of auxilliary dofs per atom
nhc_nc = 5                              # Number of integration substeps for NHCs

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
V = mdprop.init.boltzmann(kT0, masses)
X, V = mdprop.init.initialize_centered(X, V, masses)

# Construct thermostat
thermo = mdprop.update.NoseHooverNVT.build(masses, kT, damptime, chain_length=nhc_len, nc=nhc_nc)
aux_q, aux_v = mdprop.update.NoseHooverNVT.initialize(kT, thermo.params['Qs'])
state = {
    'X': X,
    'V': V,
    'symbols': symbols,
    'masses': masses,
    'aux_position_NH': aux_q,
    'aux_velocity_NH': aux_v,
}

boundary_r = mdprop.utils.density_to_spherical_radius(masses, density) * rscale
boundary_k = np.reshape(masses, (-1, 1)) * 10.0 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2 / AMU_TO_AU # a.u. = 10.0 kcal/mol/Angstrom**2/amu
boundary_pot = mdprop.potential.SoftSphere(radius=boundary_r, magnitude=boundary_k)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct update objects using pointers to different forces
    pot_list = mdprop.potential.PotentialList([DFTB, boundary_pot])
    vel_update = mdprop.update.VelocityUpdate(pot_list, masses)

    # Construct integrator (NHC / DFTB / Boundary / Position / Boundary / DFTB / NHC)
    integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
    print(integ)

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    traj.append_printkey("NHC_energy")
    traj.run(dt, sim_time, state)
