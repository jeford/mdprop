#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper
from mdprop.units import KCAL_MOL_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, AMU_TO_AU, K_TO_AU

## PARAMETERS ##
xyzfile = "../data/nitromethane20.xyz"  # XYZ file of geometry
dt = 1.0 * FS_TO_AU                     # Timestep 
nstep = 5000                            # Number of steps to take in MD
sim_time = nstep * dt                   # Time to take in MD
kT = 1000.0 * K_TO_AU                   # Thermostat temp
density = 1.14                          # Density to calculate radius of spherical piston (g/cm^3)
rscale = 0.9                            # Scaling factor of radius to increase pressure
L = 4
nc = 5
dim = 3
tau = 10.0 * FS_TO_AU
damptime = 10.0 * FS_TO_AU

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)

boundary_r = mdprop.utils.density_to_spherical_radius(masses, density) * rscale
boundary_k = np.reshape(masses, (-1, 1)) * 10.0 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2 / AMU_TO_AU # a.u. = 10.0 kcal/mol/Angstrom**2/amu
boundary_pot = mdprop.potential.SoftSphere(radius=boundary_r, magnitude=boundary_k)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    # Construct PotentialList object as a combination of the boundary and DFTB
    pot_list = mdprop.potential.PotentialList([DFTB, boundary_pot])

    # Build the integrator
    #integ = mdprop.integrator.SIN(pot_list, masses, kT, L=L, tau=tau, damptime=damptime, nc=nc, dim=dim, mass_weight=False)
    integ = mdprop.integrator.VelocityVerlet(mdprop.update.VelocityUpdate(pot_list, masses, name='potentiallist'))
    print(integ)

    #Qs = integ.updates[0].params['Qs']
    #V, aux_v = mdprop.init.sin_velocity(masses, Qs, kT)
    V = mdprop.init.boltzmann(kT, masses)

    state = {
        'X': X,
        'V': V,
        'symbols': symbols,
        'masses': masses,
        #'aux_velocity_NH': aux_v,
    }

    # Run it!
    traj = mdprop.trajectory.Trajectory(integ)
    #sin_hook = mdprop.hook.SINConstraintViolation(masses, Qs, kT)
    vt_hook = mdprop.hook.VirialTemperature(grad_key='potentiallist_gradient')
    traj.hooks.append(vt_hook)
    #traj.hooks.append(sin_hook)
    traj.append_printkeys(["virial_temperature", "constraint_violation", "potentiallist_energy"])
    traj.del_printkey("potential_energy")
    traj.run(dt, sim_time, state)
