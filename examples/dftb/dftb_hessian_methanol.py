#!/usr/bin/env python

import numpy as np

import mdprop
import mdprop.wrapper

## PARAMETERS ##
xyzfile = "../data/h2o.xyz"   # XYZ file of geometry

# Create initial state
X, symbols = mdprop.io.read_xyz(xyzfile)
masses = mdprop.utils.symbol_to_mass(symbols)
X, _ = mdprop.utils.align_principal_axes(X, masses)

# Use DFTB in with context to handle working directory for ASE
with mdprop.wrapper.DFTBPlus(symbols) as DFTB:
    H = DFTB.compute_hessian(X)
    print(H)
