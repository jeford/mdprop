import autograd.numpy as np
import autograd as ag

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mdprop

def _compute_energy(Xi):
    r = np.linalg.norm(Xi)
    return 5.0 * r + 1.0 / r

#def _compute_energy(Xi):
#    return Xi**2

_compute_gradient = ag.grad(_compute_energy)
_compute_hessian = ag.grad(_compute_gradient)

class Joukowsky(mdprop.potential.Potential):
    def __init__(self):
        pass

    def compute_energy_per_particle(self, X, **state):
        Es = np.zeros((len(X), ))
        for i, Xi in enumerate(X):
            Es[i] = _compute_energy(Xi)
        return Es

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        V = self.compute_energy(X, **state)
        Gs = np.zeros_like(X)
        for i, Xi in enumerate(X):
            Gs[i] = _compute_gradient(Xi)
        return V, Gs

    def compute_hessian(self, X, **state):
        V = self.compute_energy(X, **state)
        Gs = np.zeros_like(X)
        Hs = np.zeros_like(X)
        for i, Xi in enumerate(X):
            Gs[i] = _compute_gradient(Xi)
            Hs[i] = _compute_hessian(Xi)
        return V, Gs, Hs

pot = Joukowsky()
x = np.linspace(-2.0, -0.25, 100)
#x = np.linspace(-2.0, 2.0, 100)
u = pot.compute_energy_per_particle(x)
_, g, h = pot.compute_hessian(x)
print(x.shape)
print(u.shape)
print(g.shape)
print(h.shape)
e0 = u[0]
v = -np.sqrt(np.abs(u - e0))*np.sign(x)

l0 = (g*v)
lg = g**2
lh = v**2 * h
lt = np.abs(0.5 * lg - lh)

l0m = np.max(np.abs(l0))
lgm = np.max(np.abs(lg))
lhm = np.max(np.abs(lh))
ltm = np.max(np.abs(lt))

l0n = l0 / np.max(np.abs(l0))
lgn = lg / np.max(np.abs(lg))
lhn = lh / np.max(np.abs(lh))
ltn = lt / np.max(np.abs(lt))

def subplot(x, y, z, labely, labelz, filename):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y, color='tab:blue', label=labely)
    ax1.set_ylabel(labely, color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(x, z, color='tab:red', label=labelz)
    ax2.set_ylabel(labelz, color='tab:red')
    plt.tight_layout()
    plt.savefig(filename)
    
subplot(x, u, l0, 'Pot', 'g^T v', 'l0.eps') 
subplot(x, u, lg, 'Pot', 'g^T g', 'lg.eps') 
subplot(x, u, lh, 'Pot', 'v^T H v', 'lh.eps') 
subplot(x, u, lt, 'Pot', '0.5 g^T g - v^T H v', 'ldiff.eps') 
#plt.figure()
##plt.plot(x, u, label='potential')
##plt.plot(x, g, label='gradient')
##plt.plot(x, h, label='hessian')
#plt.plot(x, l0n, label=)
#plt.plot(x, lgn, label='lg / %f' % lgm)
#plt.plot(x, lhn, label='lh / %f' % lhm)
#plt.plot(x, ltn, label='lt / %f' % ltm)
#plt.legend()
#plt.tight_layout()
#plt.savefig("shadow.eps")
