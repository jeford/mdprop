import numpy as np

import mdprop

class DoubleWell(mdprop.potential.Potential):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute_energy_per_particle(self, X):
        X2 = X**2
        X4 = X2**2
        E = -self.a * X2 + self.b * X4
        return E
        
    def compute_gradient(self, X):
        X2 = X**2
        X4 = X2**2
        E = -self.a * np.sum(X2) + self.b * np.sum(X4)
        G = - 2.0 * self.a * X + 4.0* self.b * X2*X
        return E, G

