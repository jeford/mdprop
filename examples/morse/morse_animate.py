#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop

# Set parameters
T = 100  # Kelvin
nparticle = 3
dim = 2

# Initiatize ideal gas in 1D
#X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
X = np.random.normal(0.0, 0.25, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(T, masses, dim)
X = mdprop.init.center(X, masses)
V = mdprop.init.center(V, masses)

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'PE': 0.0,
        'aux_momentum_CN': np.zeros((2, 1*nparticle)),
        }

# Construct integrator
morse = mdprop.potential.Morse(1.0, 0.1, 0.5)
morse_update = mdprop.update.VelocityUpdate(morse.compute_forces)
integ = mdprop.integrator.VelocityVerlet(morse_update)

# Uncomment for harmonic boundary
bound = mdprop.potential.SoftSphere(0.0, 0.1)
#bound_update = mdprop.update.GeneralVelocityUpdate(bound.compute_forces)
#integ = integ.compose_into(mdprop.integrator.OneStepIntegrator(bound_update), 1.0, 0.5)

# Uncomment for thermostat
#noise = mdprop.update.WhiteNoise(T, 0.1)
#noise = update.ColoredNoise(T)
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
    state = integ.step(0.04, state)
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
