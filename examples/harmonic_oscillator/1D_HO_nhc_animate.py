#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import mdprop

# Set parameters
kT = 0.5
damping_time = 10.0
nparticle = 10
dim = 1
k = 1.0
r0 = 0.0

# Initialize particles in 1D
np.random.seed(1337)
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(kT, masses, dim)
chain_len = 3
aux_masses = np.ones((chain_len, nparticle)) * kT * damping_time**2
aux_q, aux_v = mdprop.update.NoseHooverNVT.initialize(kT, aux_masses)

state = {
        'X': X,
        'V': V,
        'masses': masses,
        'aux_position_NH': aux_q,
        'aux_velocity_NH': aux_v,
        }

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(r0, k)

# Construct updates
vel_update = mdprop.update.VelocityUpdate(bound, masses)
thermo = mdprop.update.NoseHooverNVT(masses, kT, aux_masses)

# Construct integrator
integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
print(integ)

# Initialize plots
xlim = [-2.0, 2.0]
ylim = [-0.2, 3.8]
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Left plot holds x coordinate and potential energy
ax = fig.add_subplot(121, aspect='equal', autoscale_on=True,
                     xlim=xlim, ylim=ylim)
ax2 = fig.add_subplot(121, aspect='equal', autoscale_on=True,
                     xlim=xlim, ylim=ylim)
ax.set_title('Harmonic Oscillator PE')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

# Right plot holds phase space diagram
ax3 = fig.add_subplot(122, aspect='equal', autoscale_on=False,
                     xlim=xlim, ylim=xlim)
ax4 = fig.add_subplot(122, aspect='equal', autoscale_on=False,
                     xlim=xlim, ylim=xlim)
ax3.set_title('Harmonic Oscillator Phase Space')
ax3.set_xlim(xlim)
ax3.set_ylim(xlim)
ax4.set_xlim(xlim)
ax4.set_ylim(xlim)

# Plot PES
xs = np.linspace(xlim[0], xlim[1])[:, None]
pes = bound.compute_energy_per_particle(xs)
pe = ax2.plot(xs, pes, 'k', ls='-')

# Plot analytic answer of phase space path
circles = [plt.Circle((0, 0), 0.5*i, color='k', fill=False) for i in range(5)]
for c in circles:
    ax4.add_artist(c)

# Initialize variables for holding data
particles, = ax.plot([], [], 'bo', ms=6)
phasespace, = ax3.plot([], [], 'bo', ms=6)

# Methods to initialize and update animation
def init():
    global particles, phasepace, state
    particles.set_data(state['X'][:, 0], bound.energy_per_particle(state['X'], masses=masses))
    phasespace.set_data(state['X'][:, 0], state['V'][:, 0])
    return particles, phasespace

def animate(i):
    global particles, phasespace, integ, state
    state = integ.step(0.05, state)
    particles.set_data(state['X'][:, 0], bound.energy_per_particle(state['X'], masses=masses))
    phasespace.set_data(state['X'][:, 0], state['V'][:, 0])
    return particles, phasespace

anim = animation.FuncAnimation(fig, animate,
                               frames=200, interval=1, blit=True)

plt.tight_layout()
plt.show()
