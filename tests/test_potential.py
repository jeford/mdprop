import unittest
import numpy as np

import mdprop

class TestPotential(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
                    [ 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [ 0.0, 1.0, 0.0],
                    ])
        self.V = np.array([
                    [ 0.0, 0.0, 0.0],
                    [ 0.0, 0.0, 0.0],
                    [ 1.0, 0.0, 0.0],
                    ])
        self.S = ['H', 'H', 'O']
        self.M = mdprop.utils.symbol_to_mass(self.S)

    def test_potential_list(self):
        r0 = 0.5
        k = 1.5
        sphere = mdprop.potential.SoftCube(r0, k)
        r1 = 0.25
        k1 = 2.5
        cube = mdprop.potential.SoftSphere(r1, k1)
        potential = sphere.add(cube)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.0
        for i in range(3): # Sphere
            E_check += 0.5 * k * (np.linalg.norm(self.X[i]) - r0)**2
        for i in range(3): # Cube
            for j in range(3):
                if abs(self.X[i, j]) > r1:
                    E_check += 0.5 * k1 * (abs(self.X[i, j]) - r1)**2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))

        potential = potential.add(potential)
        E, grad = potential.compute_gradient(self.X)
        E_check *= 2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))


    def test_softshalfspace(self):
        normal = np.array([0.0, 1.0, 0.0]) # Normal pointing directly toward O
        offset_mag = 0.5
        offset = np.array([0.0, offset_mag, 0.0]) # Halfspace halfway between H atoms and O atom
        k = 2.5
        potential = mdprop.potential.SoftHalfSpace(normal, k, offset)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.5 * k * (self.X[2, 1] - offset_mag)**2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))

    def test_softscube(self):
        r0 = 0.5
        k = 1.5
        potential = mdprop.potential.SoftCube(r0, k)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.0
        for i in range(3):
            for j in range(3):
                if abs(self.X[i, j]) > r0:
                    E_check += 0.5 * k * (abs(self.X[i, j]) - r0)**2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))

    def test_softsphere(self):
        r0 = 0.5
        k = 3.5
        potential = mdprop.potential.SoftSphere(r0, k)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.0
        for i in range(3):
            E_check += 0.5 * k * (np.linalg.norm(self.X[i]) - r0)**2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))

    def test_interatomicspring(self):
        ind1, ind2 = 0,2
        dist_stop = 0.5 
        k = 0.25
        potential = mdprop.potential.InteratomicSpring(ind1, ind2, dist_stop, k)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.5 * k * (np.linalg.norm(self.X[ind1] - self.X[ind2]) - dist_stop)**2
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))

    def test_classicalcoulomb(self):
        q = [1.0, 1.0, 8.0]
        k = 0.5
        potential = mdprop.potential.ClassicalCoulomb(q, k)
        E, grad = potential.compute_gradient(self.X)
        E_check = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                E_check += k * q[i] * q[j] / np.linalg.norm(self.X[i] - self.X[j])
        num_grad = mdprop.utils.numerical_gradient(self.X, potential.compute_energy)
        self.assertAlmostEqual(E, E_check)
        self.assertTrue(np.allclose(grad, num_grad))



if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPotential)
    unittest.TextTestRunner(verbosity=2).run(suite)
