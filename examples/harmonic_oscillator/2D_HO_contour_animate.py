#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

import mdprop

# Set parameters
kT = 0.25
nparticle = 20
dim = 2

# Initiatize ideal gas in 2D
X = np.random.uniform(-1.5, 1.5, (nparticle, dim))
masses = np.ones((nparticle, 1))
V = mdprop.init.boltzmann(kT, masses, dim)
state = {
        'X': X,
        'V': V,
        'masses': masses,
        'aux_momentum_CN': np.zeros((2, 1*nparticle)),
        }

# Construct harmonic oscillator potential
bound = mdprop.potential.SoftSphere(0.0, 0.35)

# Construct integrator
vel_update = mdprop.update.VelocityUpdate(bound, masses)
thermo = mdprop.update.Langevin(masses, T, damptime=1.0)
#thermo = update.ColoredNoise(masses, T)
integ = mdprop.integrator.VelocityVerletMultiple([thermo, vel_update])
print(integ)

# Initialize plots
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
xlim = [-2.0, 2.0]
ylim = [-2.0, 2.0]
zlim = [-0.2, 3.8]

ax = fig.add_subplot(111)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title('2D Harmonic Oscillator PE')

ax2 = fig.add_subplot(111)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

# Plot potential energy on grid for contour plot
gridlen = 20
xs = np.linspace(xlim[0], xlim[1], gridlen)[:, None]
ys = np.linspace(ylim[0], ylim[1], gridlen)[:, None]
xv,yv = np.meshgrid(xs, ys)
xvr = np.reshape(xv, (-1, 1))
yvr = np.reshape(yv, (-1, 1))
xys = np.hstack((xvr, yvr))
pes = bound.energy_per_particle(xys)
pes = pes.reshape(gridlen, gridlen)
pe = ax2.contour(xv, yv, pes)# 'k', ls='-')

# Initialize variables for holding data
particles, = ax.plot([], [], 'bo', ms=6)

# Methods to initialize and update animation
def init():
    global particles, state
    particles.set_data(state['X'][:, 0], state['X'][:, 1])
    return particles, 


def animate(i):
    global particles, integ, state
    state = integ.step(0.01, state)
    particles.set_data(state['X'][:, 0], state['X'][:, 1])
    return particles,

anim = animation.FuncAnimation(fig, animate,
                               frames=200, interval=1, blit=True)

plt.tight_layout()
plt.show()
