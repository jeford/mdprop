#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop

# Set parameters
kT = 0.1
damp_time = 0.1
dim = 2
nparticle = 4
dt = 0.04

# Initiatize ideal gas in 1D
X = np.random.normal(0.0, 0.25, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(kT, masses, dim)
X = mdprop.init.center(X, masses)
V = mdprop.init.center(V, masses)

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'PE': 0.0,
        'aux_momentum_CN': np.zeros((2, 1*nparticle)),
        'simulation_time': 0.00
        }

# Construct integrator for Morse fluid
morse = mdprop.potential.Morse(1.0, 0.1, 0.5)
morse_update = mdprop.update.VelocityUpdate(morse.compute_forces)
integ = mdprop.integrator.VelocityVerlet(morse_update)

# Insert distance anchor that gets interpolated
anchor = mdprop.update.DistanceAnchor(0, 1, 1.50, True)
td_anchor = mdprop.update.TimeDependent(anchor, 0.25, 1.25, 3.0)
pos_drive = mdprop.integrator.OneStepIntegrator(td_anchor)
integ = integ.insert(pos_drive, 1)

# Uncomment for harmonic boundary
bound = mdprop.potential.SoftSphere(0.0, 0.25)
bound_update = mdprop.update.GeneralVelocityUpdate(bound.compute_forces)
integ = integ.compose_into(mdprop.integrator.OneStepIntegrator(bound_update), 1.0, 0.5)

# Uncomment for thermostat
#noise = mdprop.update.WhiteNoise(kT, damp_time)
#noise = update.ColoredNoise(kT)
#integ = integ.compose_into(mdprop.integrator.OneStepIntegrator(noise), 1.0, 0.5)

integ = integ.squash()
print(integ)

# Initialize plots
xlim = [-2.0, 2.0]
ylim = [-2.0, 2.0]
fig = plt.figure()
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Plot holds x,y coordinates 
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True,
                     xlim=xlim, ylim=ylim)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Initialize variables for holding data
particles, = ax.plot([], [], 'bo', ms=6)

# Methods to initialize and update animation
def init():
    global particles, state
    particles.set_data(state['X'][:, 0], state['X'][:, 1])
    #PE = morse.compute_energy(state['X'])
    #KE = mdprop.hook.kinetic_energy.compute(state)['kinetic_energy']
    #TE = KE + PE
    return particles, 

def animate(i):
    global particles, integ, state
    state = integ.step(dt, state)
    state['simulation_time'] = state['simulation_time'] + dt
    particles.set_data(state['X'][:, 0], state['X'][:, 1])
    #PE = morse.compute_energy(state['X'])
    #KE = mdprop.hook.kinetic_energy.compute(state)['kinetic_energy']
    #TE = KE + PE
    #print(TE)
    return particles,

anim = animation.FuncAnimation(fig, animate,
                               frames=10, interval=1, blit=False)

plt.tight_layout()
plt.show()
